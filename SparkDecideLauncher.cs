using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Threading;
using System.Windows.Forms;

namespace SparkDecideInstaller
{
    public class MainForm : Form
    {
        private Button btnInstallRun;
        private Button btnOpenBrowser;
        private Button btnStopServer;
        private RichTextBox logBox;
        private ProgressBar progressBar;
        private Label statusLabel;
        private PictureBox logoBox;
        private Process serverProcess = null;
        private string workDir;
        private const string GITHUB_RAW = "https://raw.githubusercontent.com/maxpy9/spark-decide/main/";

        public MainForm()
        {
            workDir = AppDomain.CurrentDomain.BaseDirectory;
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            ServicePointManager.ServerCertificateValidationCallback = delegate { return true; };

            this.Text = "Spark-Decide — Installer & Studio Launcher";
            this.Size = new Size(760, 540);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(18, 18, 18);
            this.ForeColor = Color.FromArgb(237, 237, 237);
            this.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);
            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;

            string icoPath = Path.Combine(workDir, "spark_decide.ico");
            if (File.Exists(icoPath))
            {
                try { this.Icon = new Icon(icoPath); } catch { }
            }

            Panel headerPanel = new Panel();
            headerPanel.Dock = DockStyle.Top;
            headerPanel.Height = 86;
            headerPanel.BackColor = Color.FromArgb(24, 24, 24);
            headerPanel.Padding = new Padding(16);

            logoBox = new PictureBox();
            logoBox.Size = new Size(56, 56);
            logoBox.Location = new Point(18, 15);
            logoBox.SizeMode = PictureBoxSizeMode.Zoom;
            string pngPath = Path.Combine(workDir, "spark_decide_logo.png");
            if (File.Exists(pngPath))
            {
                try { logoBox.Image = Image.FromFile(pngPath); } catch { }
            }
            headerPanel.Controls.Add(logoBox);

            Label titleLbl = new Label();
            titleLbl.Text = "Spark-Decide — Setup & Studio Launcher";
            titleLbl.Font = new Font("Segoe UI", 14f, FontStyle.Bold);
            titleLbl.ForeColor = Color.FromArgb(62, 207, 142);
            titleLbl.AutoSize = true;
            titleLbl.Location = new Point(86, 16);
            headerPanel.Controls.Add(titleLbl);

            Label subLbl = new Label();
            subLbl.Text = "Scarica Spark-Decide da GitHub (maxpy9/spark-decide) + Rizzo Flow da HuggingFace e avvia Spark-Decide Studio";
            subLbl.Font = new Font("Segoe UI", 9f, FontStyle.Regular);
            subLbl.ForeColor = Color.FromArgb(160, 160, 160);
            subLbl.AutoSize = true;
            subLbl.Location = new Point(88, 46);
            headerPanel.Controls.Add(subLbl);

            this.Controls.Add(headerPanel);

            btnInstallRun = new Button();
            btnInstallRun.Text = "⚡ Scarica Tutto (GitHub + HF) & Avvia Spark-Decide";
            btnInstallRun.Font = new Font("Segoe UI", 10f, FontStyle.Bold);
            btnInstallRun.BackColor = Color.FromArgb(62, 207, 142);
            btnInstallRun.ForeColor = Color.FromArgb(10, 32, 22);
            btnInstallRun.FlatStyle = FlatStyle.Flat;
            btnInstallRun.FlatAppearance.BorderSize = 0;
            btnInstallRun.Size = new Size(370, 40);
            btnInstallRun.Location = new Point(18, 102);
            btnInstallRun.Cursor = Cursors.Hand;
            btnInstallRun.Click += BtnInstallRun_Click;
            this.Controls.Add(btnInstallRun);

            btnOpenBrowser = new Button();
            btnOpenBrowser.Text = "🌐 Apri Studio (localhost:7860)";
            btnOpenBrowser.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);
            btnOpenBrowser.BackColor = Color.FromArgb(34, 34, 34);
            btnOpenBrowser.ForeColor = Color.FromArgb(237, 237, 237);
            btnOpenBrowser.FlatStyle = FlatStyle.Flat;
            btnOpenBrowser.Size = new Size(210, 40);
            btnOpenBrowser.Location = new Point(398, 102);
            btnOpenBrowser.Cursor = Cursors.Hand;
            btnOpenBrowser.Click += (s, e) => Process.Start("http://127.0.0.1:7860");
            this.Controls.Add(btnOpenBrowser);

            btnStopServer = new Button();
            btnStopServer.Text = "⏹ Stop";
            btnStopServer.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);
            btnStopServer.BackColor = Color.FromArgb(48, 24, 24);
            btnStopServer.ForeColor = Color.FromArgb(250, 160, 160);
            btnStopServer.FlatStyle = FlatStyle.Flat;
            btnStopServer.Size = new Size(105, 40);
            btnStopServer.Location = new Point(618, 102);
            btnStopServer.Cursor = Cursors.Hand;
            btnStopServer.Click += BtnStopServer_Click;
            this.Controls.Add(btnStopServer);

            statusLabel = new Label();
            statusLabel.Text = "Stato: Pronto. Premi il pulsante verde per scaricare i modelli (GitHub + HuggingFace) e avviare il server.";
            statusLabel.AutoSize = false;
            statusLabel.Size = new Size(705, 22);
            statusLabel.Location = new Point(18, 152);
            statusLabel.ForeColor = Color.FromArgb(180, 180, 180);
            this.Controls.Add(statusLabel);

            progressBar = new ProgressBar();
            progressBar.Location = new Point(18, 178);
            progressBar.Size = new Size(705, 14);
            progressBar.Minimum = 0;
            progressBar.Maximum = 100;
            progressBar.Value = 0;
            this.Controls.Add(progressBar);

            logBox = new RichTextBox();
            logBox.Location = new Point(18, 202);
            logBox.Size = new Size(705, 282);
            logBox.BackColor = Color.FromArgb(12, 12, 12);
            logBox.ForeColor = Color.FromArgb(62, 207, 142);
            logBox.Font = new Font("Consolas", 9.5f, FontStyle.Regular);
            logBox.ReadOnly = true;
            logBox.BorderStyle = BorderStyle.FixedSingle;
            this.Controls.Add(logBox);

            AppendLog("=== Spark-Decide Automated Setup & Studio Launcher ===");
            AppendLog("Repo GitHub Modello Spark-Decide: https://github.com/maxpy9/spark-decide");
            AppendLog("Repo HuggingFace Base Rizzo Flow: https://huggingface.co/rizzoaiacademy/rizzo-flow");
            AppendLog("Cartella di lavoro: " + workDir);

            this.FormClosing += (s, e) => KillServer();
        }

        private void AppendLog(string text)
        {
            if (this.InvokeRequired)
            {
                this.BeginInvoke(new Action<string>(AppendLog), text);
                return;
            }
            logBox.AppendText("[" + DateTime.Now.ToString("HH:mm:ss") + "] " + text + Environment.NewLine);
            logBox.SelectionStart = logBox.Text.Length;
            logBox.ScrollToCaret();
        }

        private void SetStatus(string text, int progress)
        {
            if (this.InvokeRequired)
            {
                this.BeginInvoke(new Action<string, int>(SetStatus), text, progress);
                return;
            }
            statusLabel.Text = text;
            progressBar.Value = Math.Max(0, Math.Min(100, progress));
        }

        private void BtnInstallRun_Click(object sender, EventArgs e)
        {
            btnInstallRun.Enabled = false;
            Thread worker = new Thread(RunSetupAndLaunchPipeline);
            worker.IsBackground = true;
            worker.Start();
        }

        private void BtnStopServer_Click(object sender, EventArgs e)
        {
            KillServer();
            AppendLog("Server arrestato dall'utente.");
            SetStatus("Stato: Server fermato.", 0);
            btnInstallRun.Enabled = true;
        }

        private void KillServer()
        {
            if (serverProcess != null && !serverProcess.HasExited)
            {
                try { serverProcess.Kill(); } catch { }
                serverProcess = null;
            }
        }

        private string FindSystemPython()
        {
            string localApp = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string[] candidates = new string[] {
                Path.Combine(localApp, @"Programs\Python\Python311\python.exe"),
                Path.Combine(localApp, @"Programs\Python\Python312\python.exe"),
                Path.Combine(localApp, @"Programs\Python\Python310\python.exe"),
                @"C:\Python311\python.exe",
                @"C:\Python312\python.exe"
            };
            foreach (string c in candidates)
            {
                if (File.Exists(c)) return c;
            }
            return "python";
        }

        private int RunCommand(string fileName, string args)
        {
            AppendLog("> " + fileName + " " + args);
            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = fileName;
            psi.Arguments = args;
            psi.WorkingDirectory = workDir;
            psi.UseShellExecute = false;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.CreateNoWindow = true;

            using (Process p = new Process())
            {
                p.StartInfo = psi;
                p.OutputDataReceived += (s, ev) => { if (!string.IsNullOrEmpty(ev.Data)) AppendLog(ev.Data); };
                p.ErrorDataReceived += (s, ev) => { if (!string.IsNullOrEmpty(ev.Data)) AppendLog(ev.Data); };
                p.Start();
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                p.WaitForExit();
                return p.ExitCode;
            }
        }

        private void EnsureRepoScriptsFromGitHub()
        {
            string[] coreFiles = new string[] {
                "setup_models.py",
                "spark_decide.py",
                "snake_playground.py",
                "snake_env.py",
                "benchmark.py",
                "requirements.txt",
                "README.md",
                "spark_decide_logo.png"
            };
            using (WebClient wc = new WebClient())
            {
                wc.Headers.Add("User-Agent", "Spark-Decide-Installer/1.0");
                foreach (string f in coreFiles)
                {
                    string target = Path.Combine(workDir, f);
                    if (!File.Exists(target))
                    {
                        AppendLog("[GitHub] Scaricamento script " + f + " da maxpy9/spark-decide...");
                        wc.DownloadFile(GITHUB_RAW + f, target);
                    }
                }
            }
        }

        private void RunSetupAndLaunchPipeline()
        {
            try
            {
                // 0. Scarica i file del progetto da GitHub se l'exe viene eseguito da solo in una cartella vuota
                SetStatus("Passo 1/5: Verifica file di progetto da GitHub (maxpy9/spark-decide)...", 10);
                EnsureRepoScriptsFromGitHub();

                string venvPy = Path.Combine(workDir, @".venv\Scripts\python.exe");

                // 1. Verifica/Crea Ambiente Virtuale Python
                SetStatus("Passo 2/5: Verifica ambiente Python (.venv)...", 25);
                if (!File.Exists(venvPy))
                {
                    string sysPy = FindSystemPython();
                    AppendLog("Creazione ambiente virtuale .venv con: " + sysPy);
                    int code = RunCommand(sysPy, "-m venv .venv");
                    if (code != 0 || !File.Exists(venvPy))
                    {
                        AppendLog("Python 3.11 non trovato. Scaricamento automatico dell'installer ufficiale Python 3.11.9...");
                        string pyInstaller = Path.Combine(workDir, "python-3.11.9-amd64.exe");
                        using (WebClient wc = new WebClient())
                        {
                            wc.DownloadFile("https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe", pyInstaller);
                        }
                        AppendLog("Installazione silenziosa di Python 3.11.9 in corso...");
                        RunCommand(pyInstaller, "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0");
                        sysPy = FindSystemPython();
                        RunCommand(sysPy, "-m venv .venv");
                    }
                }
                else
                {
                    AppendLog("Ambiente virtuale .venv gia' presente.");
                }

                // 2. Verifica/Installa Dipendenze Python (PyTorch CUDA, Transformers, PEFT, BitsAndBytes, FastAPI)
                SetStatus("Passo 3/5: Verifica dipendenze Python e CUDA...", 45);
                int chk = RunCommand(venvPy, "-c \"import torch, transformers, peft, bitsandbytes, fastapi, uvicorn; print('Tutte le librerie sono installate! CUDA:', torch.cuda.is_available())\"");
                if (chk != 0)
                {
                    AppendLog("Installazione delle dipendenze necessarie (PyTorch CUDA 12.4 + librerie AI)...");
                    string trusted = "--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org";
                    RunCommand(venvPy, "-m pip install --upgrade pip " + trusted);
                    RunCommand(venvPy, "-m pip install torch --index-url https://download.pytorch.org/whl/cu124 " + trusted);
                    RunCommand(venvPy, "-m pip install \"transformers>=4.51,<5\" \"peft>=0.13,<0.18\" accelerate bitsandbytes fastapi uvicorn safetensors huggingface_hub matplotlib pandas pillow " + trusted);
                }

                // 3. Scarica Spark-Decide da GitHub e Rizzo Flow da HuggingFace
                SetStatus("Passo 4/5: Download Spark-Decide (da GitHub) e Rizzo Flow (da HuggingFace)...", 70);
                string setupScript = Path.Combine(workDir, "setup_models.py");
                if (File.Exists(setupScript))
                {
                    RunCommand(venvPy, "-u \"" + setupScript + "\"");
                }

                // 4. Avvia il Server Spark-Decide Studio
                SetStatus("Passo 5/5: Avvio di Spark-Decide Studio su http://127.0.0.1:7860...", 95);
                KillServer();

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = venvPy;
                psi.Arguments = "-u \"" + Path.Combine(workDir, "spark_decide.py") + "\"";
                psi.WorkingDirectory = workDir;
                psi.UseShellExecute = false;
                psi.RedirectStandardOutput = true;
                psi.RedirectStandardError = true;
                psi.CreateNoWindow = true;

                serverProcess = new Process();
                serverProcess.StartInfo = psi;
                serverProcess.OutputDataReceived += (s, ev) =>
                {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        AppendLog(ev.Data);
                        if (ev.Data.Contains("Uvicorn running on"))
                        {
                            SetStatus("Stato: ONLINE su http://127.0.0.1:7860", 100);
                        }
                    }
                };
                serverProcess.ErrorDataReceived += (s, ev) =>
                {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        AppendLog(ev.Data);
                        if (ev.Data.Contains("Uvicorn running on"))
                        {
                            SetStatus("Stato: ONLINE su http://127.0.0.1:7860", 100);
                        }
                    }
                };
                serverProcess.Start();
                serverProcess.BeginOutputReadLine();
                serverProcess.BeginErrorReadLine();
            }
            catch (Exception ex)
            {
                AppendLog("ERRORE: " + ex.Message);
                SetStatus("Stato: Errore durante il setup.", 0);
                this.BeginInvoke(new Action(() => btnInstallRun.Enabled = true));
            }
        }

        [STAThread]
        public static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }
}
