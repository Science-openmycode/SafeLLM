using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;

internal static class YinbianLauncher
{
    private static int Main(string[] args)
    {
        try
        {
            string launcherPath = Process.GetCurrentProcess().MainModule.FileName;
            string launcherDirectory = Path.GetDirectoryName(launcherPath);
            string productRoot = Directory.GetParent(launcherDirectory).FullName;
            string currentFile = Path.Combine(productRoot, "current.json");
            if (!File.Exists(currentFile))
                throw new FileNotFoundException("current.json is missing", currentFile);

            var serializer = new JavaScriptSerializer();
            var current = serializer.Deserialize<Dictionary<string, object>>(
                File.ReadAllText(currentFile, Encoding.UTF8));
            if (current == null || !current.ContainsKey("path"))
                throw new InvalidDataException("current.json has no path field");
            if (!current.ContainsKey("validated") || !Convert.ToBoolean(current["validated"]))
                throw new InvalidDataException("current version has not passed installation validation");

            string versionDirectory = Path.GetFullPath(Convert.ToString(current["path"]));
            string root = Path.GetPathRoot(versionDirectory);
            if (String.Equals(
                versionDirectory.TrimEnd(Path.DirectorySeparatorChar),
                root.TrimEnd(Path.DirectorySeparatorChar),
                StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("current version path cannot be a drive root");
            if (!Directory.Exists(versionDirectory))
                throw new DirectoryNotFoundException("current version directory is missing");

            string target = Path.Combine(versionDirectory, Path.GetFileName(launcherPath));
            if (!File.Exists(target))
                throw new FileNotFoundException("current program entry is missing", target);

            var start = new ProcessStartInfo(target, JoinArguments(args));
            start.WorkingDirectory = versionDirectory;
            start.UseShellExecute = false;
            Process child = Process.Start(start);
            if (child == null)
                throw new InvalidOperationException("failed to start current program version");
#if CONSOLE
            child.WaitForExit();
            return child.ExitCode;
#else
            return 0;
#endif
        }
        catch (Exception error)
        {
#if CONSOLE
            Console.Error.WriteLine("隐变智模启动失败：" + error.Message);
#else
            System.Windows.Forms.MessageBox.Show(
                "隐变智模启动失败：" + error.Message,
                "隐变智模",
                System.Windows.Forms.MessageBoxButtons.OK,
                System.Windows.Forms.MessageBoxIcon.Error);
#endif
            return 1;
        }
    }

    private static string JoinArguments(string[] args)
    {
        var output = new StringBuilder();
        foreach (string argument in args)
        {
            if (output.Length > 0) output.Append(' ');
            output.Append(QuoteArgument(argument));
        }
        return output.ToString();
    }

    private static string QuoteArgument(string argument)
    {
        if (argument.Length > 0 && argument.IndexOfAny(new[] { ' ', '\t', '\n', '\v', '"' }) < 0)
            return argument;
        var output = new StringBuilder("\"");
        int backslashes = 0;
        foreach (char character in argument)
        {
            if (character == '\\')
            {
                backslashes++;
            }
            else if (character == '"')
            {
                output.Append('\\', backslashes * 2 + 1);
                output.Append('"');
                backslashes = 0;
            }
            else
            {
                output.Append('\\', backslashes);
                output.Append(character);
                backslashes = 0;
            }
        }
        output.Append('\\', backslashes * 2);
        output.Append('"');
        return output.ToString();
    }
}
