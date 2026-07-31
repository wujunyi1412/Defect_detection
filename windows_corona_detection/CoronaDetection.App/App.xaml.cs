using System.IO;
using System.Windows;
using System.Windows.Threading;

namespace CoronaDetection;

public partial class App : Application
{
    private void Application_Startup(object sender, StartupEventArgs e)
    {
        if (e.Args.Length > 0 &&
            (e.Args[0].Equals("--self-test", StringComparison.OrdinalIgnoreCase) ||
             e.Args[0].Equals("--overlay-self-test", StringComparison.OrdinalIgnoreCase)))
        {
            Shutdown(RunSelfTest(e.Args));
            return;
        }

        MainWindow = new MainWindow();
        MainWindow.Show();
    }

    private static int RunSelfTest(string[] args)
    {
        try
        {
            if (args.Length < 2)
                throw new ArgumentException("用法：CoronaDetection.exe --self-test <图片路径>");
            string config = Path.Combine(AppContext.BaseDirectory, "config.ini");
            using var engine = new InspectionEngine();
            engine.Initialize(config);
            if (args[0].Equals("--overlay-self-test", StringComparison.OrdinalIgnoreCase))
            {
                if (args.Length < 3)
                    throw new ArgumentException(
                        "用法：CoronaDetection.exe --overlay-self-test <图片路径> <输出PNG路径>");
                NativeOverlayResult overlay = engine.ProcessToOverlayFile(
                    Path.GetFullPath(args[1]), Path.GetFullPath(args[2]));
                if (!string.IsNullOrEmpty(overlay.ExportError))
                    throw new InvalidOperationException(overlay.ExportError);
            }
            else
            {
                engine.Process(Path.GetFullPath(args[1]));
            }
            return 0;
        }
        catch (Exception ex)
        {
            File.WriteAllText(Path.Combine(AppContext.BaseDirectory, "self_test_error.log"), ex.ToString());
            return 10;
        }
    }

    private void Application_DispatcherUnhandledException(
        object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        string logPath = Path.Combine(AppContext.BaseDirectory, "startup_error.log");
        File.WriteAllText(logPath, e.Exception.ToString());
        MessageBox.Show(
            $"程序发生未处理异常：\n{e.Exception.Message}\n\n详细日志：{logPath}",
            "Corona 瑕疵检测", MessageBoxButton.OK, MessageBoxImage.Error);
        e.Handled = true;
    }
}
