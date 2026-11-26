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

    // Legacy responses and pre-classification rejections may lack a breakdown.
    private static boolean validProbability(Double value) {
        return value == null || (Double.isFinite(value) && value >= 0 && value <= 1);
    }

    private static boolean validBreakdown(AiResult output) {
        Double real = output.realProbability();
        Double fake = output.fakeProbability();
        Double invalid = output.invalidProbability();
        if (!validProbability(real) || !validProbability(fake) || !validProbability(invalid)) return false;
        if (real == null && invalid == null) return true; // old AI service or no inference
        return real != null && fake != null && invalid != null
                && Math.abs(real + fake + invalid - 1.0) <= 0.002;
    }

    public ScanRecord run(MultipartFile image, String source) throws IOException {
        if (image == null || image.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Please supply an image");
        }
        String contentType = image.getContentType();
        if (contentType == null || !SUPPORTED.contains(contentType)) {
            throw new ResponseStatusException(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "Only JPG, PNG and WebP are allowed");
        }
        if (image.getSize() > 10 * 1024 * 1024) {
            throw new ResponseStatusException(HttpStatus.PAYLOAD_TOO_LARGE, "Image is larger than 10 MB");
        }
        if (source == null || !Set.of("camera", "upload").contains(source)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid scan source");
        }
        String originalFilename = image.getOriginalFilename();
        String filename = originalFilename == null ? "capture.jpg"
                : originalFilename.replace('\\', '/').replaceAll(".*/", "");
        if (filename.isBlank()) filename = "capture.jpg";
        if (filename.length() > 200) filename = filename.substring(filename.length() - 200);
        String id = UUID.randomUUID().toString();
        final String uploadFilename = filename;
        ByteArrayResource bytes = new ByteArrayResource(image.getBytes()) {
            @Override public String getFilename() { return uploadFilename; }
        };
        HttpHeaders fileHeaders = new HttpHeaders();
        fileHeaders.setContentType(MediaType.parseMediaType(contentType));
        MultiValueMap<String, Object> payload = new LinkedMultiValueMap<>();
        payload.add("image", new HttpEntity<>(bytes, fileHeaders));
        payload.add("source", source);
        payload.add("request_id", id);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        AiResult output;
        try {
            ResponseEntity<AiResult> response = client.postForEntity(
                aiUrl + "/predict", new HttpEntity<>(payload, headers), AiResult.class);
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
                !source.equals(output.source()) || output.status() == null ||
                !Set.of("REAL", "FAKE", "REJECTED").contains(output.status()) ||
                (output.denomination() != null && (!Set.of(500, 1000).contains(output.denomination()) || "REJECTED".equals(output.status()))) ||
                !validProbability(output.confidence()) || !validBreakdown(output)) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Invalid reply from AI service");
        }
        ScanRecord record = repository.save(ScanRecord.of(output, filename));
        log.info("SCAN {} | {} | denomination={} | confidence={} | source={} | path={} | boxes={} | candidates={} | threshold={} | fakeProbability={}",
            record.getId(), record.getStatus(), record.getDenomination(), record.getConfidence(), source, record.getInferencePath(),
            record.getDetectorBoxCount(), record.getCandidateCount(), record.getThreshold(), record.getFakeProbability());
        return record;
    }

    public ResponseEntity<byte[]> historyImage(String id) {
        try {
            String safeId = UUID.fromString(id).toString();
            byte[] bytes = statusClient.getForObject(aiUrl + "/history-image/" + safeId, byte[].class);
            if (bytes == null || bytes.length == 0) {
                throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Image unavailable");
            }
            return ResponseEntity.ok().contentType(MediaType.parseMediaType("image/jpeg"))
                .header(HttpHeaders.CACHE_CONTROL, "no-store").body(bytes);
        } catch (IllegalArgumentException | HttpStatusCodeException ex) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Image unavailable");
        } catch (ResourceAccessException ex) {
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Image service unavailable");
        }
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
