package com.banknoteguard.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/** Local development origins. The web app proxies /api so CORS is not normally required. */
@Configuration
public class CorsConfig implements WebMvcConfigurer {
    @Value("${app.cors-origins:http://localhost:5174,http://127.0.0.1:5174}")
    private String origins;

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        String[] allowed = java.util.Arrays.stream(origins.split(","))
            .map(String::trim)
            .filter(item -> !item.isEmpty())
            .toArray(String[]::new);
        registry.addMapping("/api/**")
            .allowedOrigins(allowed)
            .allowedMethods("GET", "POST", "OPTIONS")
            .allowedHeaders("*")
            .maxAge(3600);
    }
}
