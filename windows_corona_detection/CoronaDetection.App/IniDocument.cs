using System.ComponentModel;
using System.IO;
using System.Text;

namespace CoronaDetection;

internal sealed class IniSetting : INotifyPropertyChanged
{
    private string _value = string.Empty;
    public required string Section { get; init; }
    public required string Key { get; init; }
    public required int LineIndex { get; init; }
    public string Comment { get; init; } = string.Empty;
    public string Value
    {
        get => _value;
        set
        {
            if (_value == value) return;
            _value = value;
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Value)));
        }
    }
    public event PropertyChangedEventHandler? PropertyChanged;
}

internal sealed class IniDocument
{
    private readonly List<string> _lines = [];
    public BindingList<IniSetting> Settings { get; } = [];

    public static IniDocument Load(string path)
    {
        var document = new IniDocument();
        document._lines.AddRange(File.ReadAllLines(path, new UTF8Encoding(false)));
        string section = string.Empty;

        for (int i = 0; i < document._lines.Count; i++)
        {
            string trimmed = document._lines[i].Trim();
            if (trimmed.StartsWith('['))
            {
                int closeBracket = trimmed.IndexOf(']');
                if (closeBracket > 1)
                {
                    string suffix = trimmed[(closeBracket + 1)..].TrimStart();
                    if (suffix.Length == 0 || suffix[0] is ';' or '#')
                    {
                        section = trimmed[1..closeBracket].Trim();
                        continue;
                    }
                }
            }
            if (trimmed.Length == 0 || trimmed[0] is ';' or '#')
                continue;

            int equals = trimmed.IndexOf('=');
            if (equals <= 0) continue;
            string key = trimmed[..equals].Trim();
            string tail = trimmed[(equals + 1)..].Trim();
            int commentAt = FindInlineComment(tail);
            string value = commentAt < 0 ? tail : tail[..commentAt].TrimEnd();
            string comment = commentAt < 0 ? string.Empty : tail[commentAt..].Trim();
            document.Settings.Add(new IniSetting
            {
                Section = section,
                Key = key,
                Value = value,
                Comment = comment,
                LineIndex = i
            });
        }
        return document;
    }

    public void Save(string path)
    {
        foreach (IniSetting setting in Settings)
        {
            string suffix = string.IsNullOrWhiteSpace(setting.Comment)
                ? string.Empty
                : "    " + setting.Comment;
            _lines[setting.LineIndex] = $"{setting.Key}={setting.Value}{suffix}";
        }
        File.WriteAllLines(path, _lines, new UTF8Encoding(false));
    }

    private static int FindInlineComment(string value)
    {
        for (int i = 1; i < value.Length; i++)
            if (value[i] is '#' or ';' && char.IsWhiteSpace(value[i - 1]))
                return i;
        return -1;
    }
}
