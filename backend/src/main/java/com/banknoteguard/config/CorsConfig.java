package com.banknoteguard.config;

import java.util.ArrayList;
import java.util.List;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.lang.NonNull;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/** Local development origins. The web app proxies /api so CORS is not normally required. */
@Configuration
public class CorsConfig implements WebMvcConfigurer {
    @Value("${app.cors-origins:http://localhost:5174,http://127.0.0.1:5174}")
    private String origins = "http://localhost:5174,http://127.0.0.1:5174";

    @Override
    public void addCorsMappings(@NonNull CorsRegistry registry) {
        List<String> cleaned = new ArrayList<>();
        for (String origin : origins.split(",")) {
            String item = origin.trim();
            if (!item.isEmpty()) {
                cleaned.add(item);
            }
        }
        String[] allowed = new String[cleaned.size()];
        cleaned.toArray(allowed);
        registry.addMapping("/api/**")
            .allowedOrigins(allowed)
            .allowedMethods("GET", "POST", "DELETE", "OPTIONS")
            .allowedHeaders("*")
            .maxAge(3600);
    }
}
