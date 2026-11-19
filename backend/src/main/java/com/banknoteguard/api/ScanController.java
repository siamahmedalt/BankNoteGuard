package com.banknoteguard.api;

import com.banknoteguard.model.ScanRecord;
import com.banknoteguard.repository.ScanRepository;
import com.banknoteguard.service.ScanService;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class ScanController {
    private final ScanService service;
    private final ScanRepository repository;

    public ScanController(ScanService service, ScanRepository repository) {
        this.service = service;
        this.repository = repository;
    }

    @PostMapping(value = "/scans", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ScanRecord scan(@RequestParam("image") MultipartFile image,
                           @RequestParam(defaultValue = "upload") String source) throws IOException {
        return service.run(image, source);
    }

    @GetMapping("/scans")
    public List<ScanRecord> history() {
        return repository.findAllByOrderByCreatedAtDesc(PageRequest.of(0, 12));
    }

    @GetMapping("/health")
    public Map<String, String> apiHealth() { return Map.of("api", "online"); }

    @GetMapping("/system")
    public Map<String, Object> health() { return service.systemStatus(); }
}
