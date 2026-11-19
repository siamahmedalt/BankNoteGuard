package com.banknoteguard.repository;

import com.banknoteguard.model.ScanRecord;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ScanRepository extends JpaRepository<ScanRecord, String> {
    List<ScanRecord> findAllByOrderByCreatedAtDesc(Pageable pageable);
}
