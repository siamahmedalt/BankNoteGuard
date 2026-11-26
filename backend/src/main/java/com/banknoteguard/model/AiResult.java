package com.banknoteguard.model;

public record AiResult(
    String requestId, String source, String status, Integer denomination,
    Double confidence, Double fakeProbability, Double threshold,
    Double detectorConfidence, String reason, Long durationMs,
    String inferencePath, Integer detectorBoxCount, Integer candidateCount,
    Double realProbability, Double invalidProbability
) {}
