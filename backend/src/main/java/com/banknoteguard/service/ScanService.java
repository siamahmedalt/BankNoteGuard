package com.banknoteguard.service;

import com.banknoteguard.model.AiResult;
import com.banknoteguard.model.ScanRecord;
import com.banknoteguard.repository.ScanRepository;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

@Service
public class ScanService {
    private static final Logger log = LoggerFactory.getLogger(ScanService.class);
    private static final Set<String> SUPPORTED = Set.of("image/jpeg", "image/png", "image/webp");
    private final ScanRepository repository;
    private final RestTemplate client = buildClient(10_000, 300_000);
    private final RestTemplate statusClient = buildClient(2_000, 5_000);

    private static RestTemplate buildClient(int connectMs, int readMs) {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(connectMs);
        requestFactory.setReadTimeout(readMs);
        return new RestTemplate(requestFactory);
    }
    private final String aiUrl;

    public ScanService(ScanRepository repository, @Value("${ai.base-url}") String aiUrl) {
        this.repository = repository;
        this.aiUrl = aiUrl;
    }

    public ScanRecord run(MultipartFile image, String source) throws IOException {
        if (image == null || image.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Please supply an image");
        }
        if (!SUPPORTED.contains(image.getContentType())) {
            throw new ResponseStatusException(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "Only JPG, PNG and WebP are allowed");
        }
        if (image.getSize() > 10 * 1024 * 1024) {
            throw new ResponseStatusException(HttpStatus.PAYLOAD_TOO_LARGE, "Image is larger than 10 MB");
        }
        if (!Set.of("camera", "upload").contains(source)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid scan source");
        }
        String filename = image.getOriginalFilename() == null ? "capture.jpg"
                : image.getOriginalFilename().replace('\\', '/').replaceAll(".*/", "");
        if (filename.length() > 200) filename = filename.substring(filename.length() - 200);
        String id = UUID.randomUUID().toString();
        final String uploadFilename = filename;
        ByteArrayResource bytes = new ByteArrayResource(image.getBytes()) {
            @Override public String getFilename() { return uploadFilename; }
        };
        HttpHeaders fileHeaders = new HttpHeaders();
        fileHeaders.setContentType(MediaType.parseMediaType(image.getContentType()));
        MultiValueMap<String, Object> payload = new LinkedMultiValueMap<>();
        payload.add("image", new HttpEntity<>(bytes, fileHeaders));
        payload.add("source", source);
        payload.add("request_id", id);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        AiResult output;
        try {
            ResponseEntity<AiResult> response = client.exchange(
                aiUrl + "/predict", HttpMethod.POST, new HttpEntity<>(payload, headers), AiResult.class);
            output = response.getBody();
        } catch (HttpStatusCodeException ex) {
            log.warn("AI service rejected scan {} with status {}: {}", id, ex.getStatusCode(), ex.getResponseBodyAsString());
            HttpStatus status = HttpStatus.resolve(ex.getStatusCode().value());
            String detail = ex.getResponseBodyAsString();
            try {
                var node = new com.fasterxml.jackson.databind.ObjectMapper().readTree(detail);
                detail = node.path("detail").asText(detail);
            } catch (Exception ignored) {}
            throw new ResponseStatusException(status != null ? status : HttpStatus.BAD_GATEWAY, detail);
        } catch (ResourceAccessException ex) {
            log.error("Python inference endpoint unreachable: {}", aiUrl, ex);
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                "AI service is unreachable inside Docker at " + aiUrl + ". Check docker compose logs ai.");
        }
        if (output == null || !id.equals(output.requestId()) ||
                !Set.of("REAL", "FAKE", "REJECTED").contains(output.status())) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Invalid reply from AI service");
        }
        ScanRecord record = repository.save(ScanRecord.of(output, filename));
        log.info("SCAN {} | {} | denomination={} | confidence={} | source={}",
            record.getId(), record.getStatus(), record.getDenomination(), record.getConfidence(), source);
        return record;
    }

    public Map<String, Object> systemStatus() {
        try {
            @SuppressWarnings("unchecked") Map<String, Object> ai = statusClient.getForObject(aiUrl + "/health", Map.class);
            return Map.of("api", "online", "ai", ai != null ? ai : Map.of(
                "ready", false, "message", "AI returned an empty health response"));
        } catch (Exception ex) {
            log.warn("AI health probe at {} failed: {}", aiUrl, ex.toString());
            return Map.of("api", "online", "ai", Map.of(
                "ready", false,
                "loading", false,
                "service", "unreachable",
                "message", "Python AI is not reachable through the Docker network"));
        }
    }
}
