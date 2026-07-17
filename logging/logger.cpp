#include "logger.h"

#include <fstream>
#include <iostream>

namespace InspectionLogging {
namespace {

std::mutex g_logger_mutex;
LoggerConfig g_logger_config;

// 判断是否应该记录日志
bool ShouldLog(LogLevel level, const LoggerConfig& config) {
    return config.enabled && static_cast<int>(level) <= static_cast<int>(config.min_level);
}

}  // namespace

// 设置日志配置
void SetLoggerConfig(const LoggerConfig& config) {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    g_logger_config = config;
}

// 获取日志配置
LoggerConfig GetLoggerConfig() {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    return g_logger_config;
}

// 根据日志级别打印/记录日志消息
void LogMessage(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    if (!ShouldLog(level, g_logger_config)) {
        return;
    }

    const std::string line = "[" + std::string(ToString(level)) + "] " + message;
    if (g_logger_config.log_to_stderr) {
        std::cerr << line << std::endl;
    }

    if (g_logger_config.log_to_file && !g_logger_config.file_path.empty()) {
        std::ofstream ofs(g_logger_config.file_path, std::ios::app);
        if (ofs.is_open()) {
            ofs << line << '\n';
        }
    }
}

// 将日志级别转换为字符串
const char* ToString(LogLevel level) {
    switch (level) {
    case LogLevel::Error:
        return "ERROR";
    case LogLevel::Warning:
        return "WARN";
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Debug:
        return "DEBUG";
    default:
        return "UNKNOWN";
    }
}

// 尝试将字符串解析为日志级别
bool TryParseLogLevel(const std::string& text, LogLevel& out_level) {
    if (text == "error" || text == "ERROR" || text == "Error") {
        out_level = LogLevel::Error;
        return true;
    }
    if (text == "warning" || text == "WARN" || text == "warn" || text == "Warning") {
        out_level = LogLevel::Warning;
        return true;
    }
    if (text == "info" || text == "INFO" || text == "Info") {
        out_level = LogLevel::Info;
        return true;
    }
    if (text == "debug" || text == "DEBUG" || text == "Debug") {
        out_level = LogLevel::Debug;
        return true;
    }
    return false;
}

}  // namespace InspectionLogging
