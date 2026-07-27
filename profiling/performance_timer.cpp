#include "performance_timer.h"

#include <iomanip>
#include <sstream>

namespace InspectionProfiling {

PerformanceTimer::PerformanceTimer(bool enabled)
    : enabled_(enabled),
      start_(enabled ? Clock::now() : Clock::time_point{}) {}

void PerformanceTimer::Reset() {
    if (enabled_) {
        start_ = Clock::now();
    }
}

double PerformanceTimer::ElapsedMilliseconds() const {
    if (!enabled_) {
        return 0.0;
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
}

double PerformanceTimer::RestartMilliseconds() {
    if (!enabled_) {
        return 0.0;
    }

    const auto now = Clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(now - start_).count();
    start_ = now;
    return elapsed_ms;
}

bool PerformanceTimer::IsEnabled() const {
    return enabled_;
}

std::string FormatTimingLog(std::initializer_list<TimingEntry> entries) {
    std::ostringstream timing;
    timing << std::fixed << std::setprecision(3) << "timing_ms";
    for (const auto& entry : entries) {
        timing << ' ' << entry.name << '=' << entry.elapsed_ms;
    }
    return timing.str();
}

std::string InferenceTiming::ToLogString() const {
    return FormatTimingLog({
        {"preprocess", preprocess_ms},
        {"yolo_infer", yolo_infer_ms},
        {"patchcore_infer", patchcore_infer_ms},
        {"models_wall", models_wall_ms},
        {"patchcore_post", patchcore_post_ms},
        {"yolo_post", yolo_post_ms},
        {"compose", compose_ms},
        {"total", total_ms},
    });
}

}  // namespace InspectionProfiling
