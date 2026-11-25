package com.banknoteguard.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;
import org.springframework.lang.NonNull;

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
    private String inferencePath;
    private Integer detectorBoxCount;
    private Integer candidateCount;
    @Column(nullable = false)
    private String source;
    private String originalFilename;
    @Column(length = 600)
    private String reason;

    protected ScanRecord() {}

    @NonNull
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
        record.inferencePath = result.inferencePath();
        record.detectorBoxCount = result.detectorBoxCount();
        record.candidateCount = result.candidateCount();
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
    public String getInferencePath() { return inferencePath; }
    public Integer getDetectorBoxCount() { return detectorBoxCount; }
    public Integer getCandidateCount() { return candidateCount; }
    public String getSource() { return source; }
    public String getOriginalFilename() { return originalFilename; }
    public String getReason() { return reason; }
}
