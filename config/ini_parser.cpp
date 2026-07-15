#include "ini_parser.h"

#include <algorithm>
#include <cctype>
#include <fstream>

namespace InspectionConfig {
namespace {

// 去除字符串首尾空白字符
std::string Trim(std::string s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

// 将字符串转为小写
std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

// 判断字符串 s 是否以 prefix 开头
bool StartsWith(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

// 判断路径是否为绝对路径（盘符、UNC、/ 或 \ 开头）
bool IsAbsolutePath(const std::string& s) {
    if (s.size() >= 2 && std::isalpha(static_cast<unsigned char>(s[0])) && s[1] == ':') return true;
    if (StartsWith(s, "\\\\")) return true;
    if (!s.empty() && (s[0] == '/' || s[0] == '\\')) return true;
    return false;
}

// 获取路径的父目录部分
std::string DirnameOf(const std::string& path) {
    const auto pos = path.find_last_of("\\/");
    if (pos == std::string::npos) return std::string();
    return path.substr(0, pos);
}

// 拼接目录和路径
std::string JoinPath(const std::string& dir, const std::string& path) {
    if (dir.empty()) return path;
    if (path.empty()) return dir;
    const char last = dir.back();
    if (last == '\\' || last == '/') return dir + path;
    return dir + "\\" + path;
}

}  // namespace

// 解析 INI 文件，将结果存入 IniData 结构
// 支持 ; 和 # 开头的注释行，以及行内注释
bool ParseIniFile(const std::string& file_path, IniData& out, std::string& err) {
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        err = "open ini failed: " + file_path;
        return false;
    }

    std::string section;
    std::string line;
    while (std::getline(ifs, line)) {
        line = Trim(line);
        if (line.empty()) continue;
        if (line[0] == ';' || line[0] == '#') continue;

        for (size_t i = 0; i < line.size(); ++i) {
            const char ch = line[i];
            if (ch != ';' && ch != '#') continue;
            if (i > 0 && !std::isspace(static_cast<unsigned char>(line[i - 1]))) continue;
            line = Trim(line.substr(0, i));
            break;
        }
        if (line.empty()) continue;

        if (line.front() == '[' && line.back() == ']') {
            section = ToLower(Trim(line.substr(1, line.size() - 2)));
            continue;
        }

        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = ToLower(Trim(line.substr(0, eq)));
        std::string value = Trim(line.substr(eq + 1));
        if (key.empty()) continue;
        out.sections[section][key] = value;
    }

    return true;
}

// 从 IniData 中查找指定 section 下的 key，大小写不敏感，找到返回 true 并写入 out
bool TryGetString(const IniData& ini, const std::string& section, const std::string& key, std::string& out) {
    const std::string sec = ToLower(section);
    const std::string lowered_key = ToLower(key);
    auto it_sec = ini.sections.find(sec);
    if (it_sec == ini.sections.end()) return false;
    auto it_key = it_sec->second.find(lowered_key);
    if (it_key == it_sec->second.end()) return false;
    out = it_key->second;
    return true;
}

// 查找字符串值，找不到则返回默认值 def
std::string GetStringOr(const IniData& ini, const std::string& section, const std::string& key, const std::string& def) {
    std::string value;
    if (!TryGetString(ini, section, key, value)) return def;
    return value;
}

// 查找 float 值，找到返回 true 并写入 out
bool TryGetFloat(const IniData& ini, const std::string& section, const std::string& key, float& out) {
    std::string value;
    if (!TryGetString(ini, section, key, value)) return false;
    try {
        out = std::stof(value);
        return true;
    } catch (...) {
        return false;
    }
}

// 查找 float 值，找不到或解析失败则返回默认值 def
float GetFloatOr(const IniData& ini, const std::string& section, const std::string& key, float def) {
    float value = 0.0f;
    if (!TryGetFloat(ini, section, key, value)) return def;
    return value;
}

// 查找 int 值，找到返回 true 并写入 out
bool TryGetInt(const IniData& ini, const std::string& section, const std::string& key, int& out) {
    std::string value;
    if (!TryGetString(ini, section, key, value)) return false;
    try {
        out = std::stoi(value);
        return true;
    } catch (...) {
        return false;
    }
}

// 查找 int 值，找不到或解析失败则返回默认值 def
int GetIntOr(const IniData& ini, const std::string& section, const std::string& key, int def) {
    int value = 0;
    if (!TryGetInt(ini, section, key, value)) return def;
    return value;
}

// 查找 bool 值，支持 1/0、true/false、yes/no、y/n，找不到或无法识别则返回默认值 def
bool GetBoolOr(const IniData& ini, const std::string& section, const std::string& key, bool def) {
    std::string value;
    if (!TryGetString(ini, section, key, value)) return def;
    std::string lowered = ToLower(value);
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "y") return true;
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "n") return false;
    return def;
}

// 将相对路径转换为基于 INI 文件所在目录的绝对路径，已是绝对路径则原样返回
std::string ResolvePathRelativeToIni(const std::string& ini_path, const std::string& path) {
    if (path.empty() || IsAbsolutePath(path)) return path;
    return JoinPath(DirnameOf(ini_path), path);
}

}  // namespace InspectionConfig
