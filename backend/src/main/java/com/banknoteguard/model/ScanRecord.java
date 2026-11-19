package com.banknoteguard.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;

@Entity
@Table(name = "scans")
public class ScanRecord {
    @Id
    private String id;
    @Column(nullable = false)
    private Instant createdAt;
    @Column(nullable = false)
    private String status;
    private Integer denomination;
    private Double confidence;
    private Double fakeProbability;
    private Double threshold;
    private Double detectorConfidence;
    private Long durationMs;
    @Column(nullable = false)
    private String source;
    private String originalFilename;
    @Column(length = 600)
    private String reason;

    protected ScanRecord() {}

    public static ScanRecord of(AiResult result, String filename) {
        ScanRecord record = new ScanRecord();
        record.id = result.requestId();
        record.createdAt = Instant.now();
        record.status = result.status();
        record.denomination = result.denomination();
        record.confidence = result.confidence();
        record.fakeProbability = result.fakeProbability();
        record.threshold = result.threshold();
        record.detectorConfidence = result.detectorConfidence();
        record.durationMs = result.durationMs();
        record.source = result.source();
        record.originalFilename = filename;
        record.reason = result.reason();
        return record;
    }

    public String getId() { return id; }
    public Instant getCreatedAt() { return createdAt; }
    public String getStatus() { return status; }
    public Integer getDenomination() { return denomination; }
    public Double getConfidence() { return confidence; }
    public Double getFakeProbability() { return fakeProbability; }
    public Double getThreshold() { return threshold; }
    public Double getDetectorConfidence() { return detectorConfidence; }
    public Long getDurationMs() { return durationMs; }
    public String getSource() { return source; }
    public String getOriginalFilename() { return originalFilename; }
    public String getReason() { return reason; }
}
