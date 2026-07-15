#pragma once

#include <string>
#include <unordered_map>

namespace InspectionConfig {

struct IniData {
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> sections;
};

bool ParseIniFile(const std::string& file_path, IniData& out, std::string& err);
bool TryGetString(const IniData& ini, const std::string& section, const std::string& key, std::string& out);
std::string GetStringOr(const IniData& ini, const std::string& section, const std::string& key, const std::string& def);
bool TryGetFloat(const IniData& ini, const std::string& section, const std::string& key, float& out);
float GetFloatOr(const IniData& ini, const std::string& section, const std::string& key, float def);
bool TryGetInt(const IniData& ini, const std::string& section, const std::string& key, int& out);
int GetIntOr(const IniData& ini, const std::string& section, const std::string& key, int def);
bool GetBoolOr(const IniData& ini, const std::string& section, const std::string& key, bool def);
std::string ResolvePathRelativeToIni(const std::string& ini_path, const std::string& path);

}  // namespace InspectionConfig
