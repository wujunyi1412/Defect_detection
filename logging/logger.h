#pragma once

#include <mutex>
#include <string>

namespace InspectionLogging {

enum class LogLevel {
    Error = 0,
    Warning = 1,
    Info = 2,
    Debug = 3,
};

struct LoggerConfig {
    bool enabled = true;
    LogLevel min_level = LogLevel::Info;
    bool log_to_stderr = true;
    bool log_to_file = false;
    std::string file_path;
};

void SetLoggerConfig(const LoggerConfig& config);
LoggerConfig GetLoggerConfig();

void LogMessage(LogLevel level, const std::string& message);
const char* ToString(LogLevel level);
bool TryParseLogLevel(const std::string& text, LogLevel& out_level);

}  // namespace InspectionLogging
