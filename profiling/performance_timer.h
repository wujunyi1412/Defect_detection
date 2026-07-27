#pragma once

#include <chrono>
#include <initializer_list>
#include <string>

namespace InspectionProfiling {

class PerformanceTimer {
public:
    explicit PerformanceTimer(bool enabled = true);

    void Reset();
    double ElapsedMilliseconds() const;
    double RestartMilliseconds();
    bool IsEnabled() const;

private:
    using Clock = std::chrono::steady_clock;

    bool enabled_;
    Clock::time_point start_;
};

struct TimingEntry {
    const char* name;
    double elapsed_ms;
};

std::string FormatTimingLog(std::initializer_list<TimingEntry> entries);

struct InferenceTiming {
    double preprocess_ms = 0.0;
    double yolo_infer_ms = 0.0;
    double patchcore_infer_ms = 0.0;
    double models_wall_ms = 0.0;
    double patchcore_post_ms = 0.0;
    double yolo_post_ms = 0.0;
    double compose_ms = 0.0;
    double total_ms = 0.0;

    std::string ToLogString() const;
};

}  // namespace InspectionProfiling
