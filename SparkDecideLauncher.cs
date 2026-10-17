using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Forms;

namespace SparkDecideInstaller
{
    public class MainForm : Form
    {
        private Button btnInstallAll;
        private Button btnStartServer;
        private Button btnStopServer;
        private Button btnOpenBrowser;
        private RichTextBox logBox;
        private ProgressBar progressBar;
        private Label statusLabel;
        private PictureBox logoBox;
        private Process serverProcess = null;
        private Process tunnelProcess = null;
        private string workDir;
        private string activeTunnelUrl = "";
        private const string GITHUB_RAW = "https://raw.githubusercontent.com/maxpy9/spark-decide/main/";

        public MainForm()
        {
            workDir = AppDomain.CurrentDomain.BaseDirectory;
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            ServicePointManager.ServerCertificateValidationCallback = delegate { return true; };

            this.Text = "Spark-Decide — Installer & Studio Launcher";
            this.Size = new Size(820, 580);
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

            // Top Header Banner
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
            subLbl.Text = "Installatore rapido dei modelli & Launcher del server per PC e Telefono (con Cloudflare Tunnel)";
            subLbl.Font = new Font("Segoe UI", 9f, FontStyle.Regular);
            subLbl.ForeColor = Color.FromArgb(160, 160, 160);
            subLbl.AutoSize = true;
            subLbl.Location = new Point(88, 46);
            headerPanel.Controls.Add(subLbl);

            this.Controls.Add(headerPanel);

            // 1. Tasto "Installa tutto"
            btnInstallAll = new Button();
            btnInstallAll.Text = "📦 Installa tutto";
            btnInstallAll.Font = new Font("Segoe UI", 9.5f, FontStyle.Bold);
            btnInstallAll.BackColor = Color.FromArgb(40, 40, 40);
            btnInstallAll.ForeColor = Color.FromArgb(237, 237, 237);
            btnInstallAll.FlatStyle = FlatStyle.Flat;
            btnInstallAll.FlatAppearance.BorderColor = Color.FromArgb(80, 80, 80);
            btnInstallAll.Size = new Size(180, 42);
            btnInstallAll.Location = new Point(18, 102);
            btnInstallAll.Cursor = Cursors.Hand;
            btnInstallAll.Click += BtnInstallAll_Click;
            this.Controls.Add(btnInstallAll);

            // 2. Tasto "Esegui server"
            btnStartServer = new Button();
            btnStartServer.Text = "▶ Esegui server";
            btnStartServer.Font = new Font("Segoe UI", 10f, FontStyle.Bold);
            btnStartServer.BackColor = Color.FromArgb(62, 207, 142);
            btnStartServer.ForeColor = Color.FromArgb(10, 32, 22);
            btnStartServer.FlatStyle = FlatStyle.Flat;
            btnStartServer.FlatAppearance.BorderSize = 0;
            btnStartServer.Size = new Size(190, 42);
            btnStartServer.Location = new Point(208, 102);
            btnStartServer.Cursor = Cursors.Hand;
            btnStartServer.Click += BtnStartServer_Click;
            this.Controls.Add(btnStartServer);

            // 3. Tasto "Stoppa il server"
            btnStopServer = new Button();
            btnStopServer.Text = "⏹ Stoppa il server";
            btnStopServer.Font = new Font("Segoe UI", 9.5f, FontStyle.Bold);
            btnStopServer.BackColor = Color.FromArgb(48, 24, 24);
            btnStopServer.ForeColor = Color.FromArgb(250, 160, 160);
            btnStopServer.FlatStyle = FlatStyle.Flat;
            btnStopServer.FlatAppearance.BorderColor = Color.FromArgb(90, 40, 40);
            btnStopServer.Size = new Size(185, 42);
            btnStopServer.Location = new Point(408, 102);
            btnStopServer.Cursor = Cursors.Hand;
            btnStopServer.Enabled = false;
            btnStopServer.Click += BtnStopServer_Click;
            this.Controls.Add(btnStopServer);

            // 4. Tasto "Apri Studio"
            btnOpenBrowser = new Button();
            btnOpenBrowser.Text = "🌐 Apri Studio";
            btnOpenBrowser.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);
            btnOpenBrowser.BackColor = Color.FromArgb(28, 28, 28);
            btnOpenBrowser.ForeColor = Color.FromArgb(200, 200, 200);
            btnOpenBrowser.FlatStyle = FlatStyle.Flat;
            btnOpenBrowser.FlatAppearance.BorderColor = Color.FromArgb(60, 60, 60);
            btnOpenBrowser.Size = new Size(175, 42);
            btnOpenBrowser.Location = new Point(603, 102);
            btnOpenBrowser.Cursor = Cursors.Hand;
            btnOpenBrowser.Click += (s, e) => {
                string url = !string.IsNullOrEmpty(activeTunnelUrl) ? activeTunnelUrl : "http://127.0.0.1:7860";
                Process.Start(url);
            };
            this.Controls.Add(btnOpenBrowser);

            // Status label
            statusLabel = new Label();
            statusLabel.Text = "Stato: Pronto. Clicca 'Installa tutto' per verificare i file o 'Esegui server' per avviare subito.";
            statusLabel.AutoSize = false;
            statusLabel.Size = new Size(760, 22);
            statusLabel.Location = new Point(18, 154);
            statusLabel.ForeColor = Color.FromArgb(180, 180, 180);
            this.Controls.Add(statusLabel);

            // Progress bar
            progressBar = new ProgressBar();
            progressBar.Location = new Point(18, 178);
            progressBar.Size = new Size(760, 14);
            progressBar.Minimum = 0;
            progressBar.Maximum = 100;
            progressBar.Value = 0;
            this.Controls.Add(progressBar);

            // Log console
            logBox = new RichTextBox();
            logBox.Location = new Point(18, 202);
            logBox.Size = new Size(760, 320);
            logBox.BackColor = Color.FromArgb(12, 12, 12);
            logBox.ForeColor = Color.FromArgb(62, 207, 142);
            logBox.Font = new Font("Consolas", 9.5f, FontStyle.Regular);
            logBox.ReadOnly = true;
            logBox.BorderStyle = BorderStyle.FixedSingle;
            this.Controls.Add(logBox);

            AppendLog("=== Spark-Decide Automated Setup & Studio Launcher ===");
            AppendLog("Modello Spark-Decide (GitHub): https://github.com/maxpy9/spark-decide");
            AppendLog("Modello Foundation Rizzo Flow (HF): https://huggingface.co/rizzoaiacademy/rizzo-flow");
            AppendLog("Cartella di lavoro: " + workDir);

            this.FormClosing += (s, e) => KillProcesses();
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

        private void BtnInstallAll_Click(object sender, EventArgs e)
        {
            btnInstallAll.Enabled = false;
            btnStartServer.Enabled = false;
            Thread worker = new Thread(RunInstallPipelineOnly);
            worker.IsBackground = true;
            worker.Start();
        }

        private void BtnStartServer_Click(object sender, EventArgs e)
        {
            btnStartServer.Enabled = false;
            btnStopServer.Enabled = true;
            Thread worker = new Thread(RunServerPipeline);
            worker.IsBackground = true;
            worker.Start();
        }

        private void BtnStopServer_Click(object sender, EventArgs e)
        {
            KillProcesses();
            AppendLog("Server e Tunnel arrestati con successo.");
            SetStatus("Stato: Server fermato.", 0);
            btnStartServer.Enabled = true;
            btnStopServer.Enabled = false;
        }

        private void KillProcesses()
        {
            if (serverProcess != null && !serverProcess.HasExited)
            {
                try { serverProcess.Kill(); } catch { }
                serverProcess = null;
            }
            if (tunnelProcess != null && !tunnelProcess.HasExited)
            {
                try { tunnelProcess.Kill(); } catch { }
                tunnelProcess = null;
            }
            // Cleanup cloudflared child processes if any
            try
            {
                foreach (Process p in Process.GetProcessesByName("cloudflared"))
                {
                    try { p.Kill(); } catch { }
                }
            }
            catch { }
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

        private void EnsureRepoScripts()
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
                wc.Headers.Add("User-Agent", "Spark-Decide-Installer/2.0");
                foreach (string f in coreFiles)
                {
                    string target = Path.Combine(workDir, f);
                    if (!File.Exists(target))
                    {
                        AppendLog("[GitHub] Scaricamento file di progetto " + f + "...");
                        wc.DownloadFile(GITHUB_RAW + f, target);
                    }
                }
            }
        }

        private bool EnsurePythonAndDeps(string venvPy)
        {
            if (!File.Exists(venvPy))
            {
                string sysPy = FindSystemPython();
                AppendLog("Creazione rapida ambiente virtuale .venv con: " + sysPy);
                int code = RunCommand(sysPy, "-m venv .venv");
                if (code != 0 || !File.Exists(venvPy))
                {
                    AppendLog("Python di sistema non trovato. Download automatico di Python 3.11.9...");
                    string pyInstaller = Path.Combine(workDir, "python-3.11.9-amd64.exe");
                    using (WebClient wc = new WebClient())
                    {
                        wc.DownloadFile("https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe", pyInstaller);
                    }
                    AppendLog("Installazione di Python 3.11.9...");
                    RunCommand(pyInstaller, "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0");
                    sysPy = FindSystemPython();
                    RunCommand(sysPy, "-m venv .venv");
                }
            }

            // Verifica istantanea se le librerie sono gia' presenti
            int chk = RunCommand(venvPy, "-c \"import torch, transformers, peft, bitsandbytes, fastapi, uvicorn; print('Librerie Python gia\\' pronte! CUDA:', torch.cuda.is_available())\"");
            if (chk == 0)
            {
                AppendLog("Dipendenze Python gia' installate e integre (salto pip).");
                return true;
            }

            AppendLog("Installazione veloce delle dipendenze mancanti...");
            string trusted = "--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org";
            RunCommand(venvPy, "-m pip install torch --index-url https://download.pytorch.org/whl/cu124 " + trusted);
            RunCommand(venvPy, "-m pip install \"transformers>=4.51,<5\" \"peft>=0.13,<0.18\" accelerate bitsandbytes fastapi uvicorn safetensors huggingface_hub matplotlib pandas pillow " + trusted);
            return true;
        }

        private void RunInstallPipelineOnly()
        {
            try
            {
                SetStatus("1/3: Verifica file progetto...", 20);
                EnsureRepoScripts();

                string venvPy = Path.Combine(workDir, @".venv\Scripts\python.exe");
                SetStatus("2/3: Verifica ambiente Python e CUDA...", 50);
                EnsurePythonAndDeps(venvPy);

                SetStatus("3/3: Verifica e download modelli (GitHub + HuggingFace)...", 80);
                string setupScript = Path.Combine(workDir, "setup_models.py");
                if (File.Exists(setupScript))
                {
                    RunCommand(venvPy, "-u \"" + setupScript + "\"");
                }

                SetStatus("Stato: Installazione completata con successo!", 100);
                AppendLog("=================================================");
                AppendLog("INSTALLAZIONE COMPLETATA! Ora clicca 'Esegui server'.");
                AppendLog("=================================================");
            }
            catch (Exception ex)
            {
                AppendLog("ERRORE durante l'installazione: " + ex.Message);
                SetStatus("Stato: Errore nell'installazione.", 0);
            }
            finally
            {
                this.BeginInvoke(new Action(() => {
                    btnInstallAll.Enabled = true;
                    btnStartServer.Enabled = true;
                }));
            }
        }

        private void RunServerPipeline()
        {
            try
            {
                KillProcesses();
                string venvPy = Path.Combine(workDir, @".venv\Scripts\python.exe");

                if (!File.Exists(venvPy))
                {
                    AppendLog("Ambiente .venv non trovato. Eseguo installazione preliminare...");
                    RunInstallPipelineOnly();
                }

                SetStatus("Avvio server Spark-Decide...", 50);

                // 1. Avvia server FastAPI
                ProcessStartInfo psiServer = new ProcessStartInfo();
                psiServer.FileName = venvPy;
                psiServer.Arguments = "-u \"" + Path.Combine(workDir, "spark_decide.py") + "\" --no-browser --port 7860";
                psiServer.WorkingDirectory = workDir;
                psiServer.UseShellExecute = false;
                psiServer.RedirectStandardOutput = true;
                psiServer.RedirectStandardError = true;
                psiServer.CreateNoWindow = true;

                serverProcess = new Process();
                serverProcess.StartInfo = psiServer;
                serverProcess.OutputDataReceived += (s, ev) => {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        AppendLog(ev.Data);
                        if (ev.Data.Contains("Uvicorn running on"))
                        {
                            SetStatus("🟢 Server ATTIVO su http://127.0.0.1:7860", 100);
                        }
                    }
                };
                serverProcess.ErrorDataReceived += (s, ev) => {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        AppendLog(ev.Data);
                        if (ev.Data.Contains("Uvicorn running on"))
                        {
                            SetStatus("🟢 Server ATTIVO su http://127.0.0.1:7860", 100);
                        }
                    }
                };
                serverProcess.Start();
                serverProcess.BeginOutputReadLine();
                serverProcess.BeginErrorReadLine();

                // 2. Avvia Cloudflare Tunnel per accesso da Telefono / GitHub Pages
                string cfExe = Path.Combine(workDir, "cloudflared.exe");
                if (File.Exists(cfExe))
                {
                    AppendLog("Avvio Cloudflare Tunnel per accesso da Telefono / GitHub Pages...");
                    ProcessStartInfo psiTunnel = new ProcessStartInfo();
                    psiTunnel.FileName = cfExe;
                    psiTunnel.Arguments = "tunnel --url http://127.0.0.1:7860 --no-autoupdate";
                    psiTunnel.WorkingDirectory = workDir;
                    psiTunnel.UseShellExecute = false;
                    psiTunnel.RedirectStandardOutput = true;
                    psiTunnel.RedirectStandardError = true;
                    psiTunnel.CreateNoWindow = true;

                    tunnelProcess = new Process();
                    tunnelProcess.StartInfo = psiTunnel;
                    tunnelProcess.OutputDataReceived += (s, ev) => HandleTunnelOutput(ev.Data);
                    tunnelProcess.ErrorDataReceived += (s, ev) => HandleTunnelOutput(ev.Data);
                    tunnelProcess.Start();
                    tunnelProcess.BeginOutputReadLine();
                    tunnelProcess.BeginErrorReadLine();
                }
            }
            catch (Exception ex)
            {
                AppendLog("ERRORE durante l'avvio del server: " + ex.Message);
                SetStatus("Stato: Errore avvio server.", 0);
                this.BeginInvoke(new Action(() => {
                    btnStartServer.Enabled = true;
                    btnStopServer.Enabled = false;
                }));
            }
        }

        private void HandleTunnelOutput(string line)
        {
            if (string.IsNullOrEmpty(line)) return;
            Match m = Regex.Match(line, @"https://[a-zA-Z0-9-]+\.trycloudflare\.com");
            if (m.Success)
            {
                string foundUrl = m.Value;
                activeTunnelUrl = foundUrl;
                AppendLog("=================================================");
                AppendLog("🌐 URL PUBBLICO ATTIVO PER TELEFONO / GITHUB PAGES:");
                AppendLog(foundUrl);
                AppendLog("=================================================");
                try
                {
                    File.WriteAllText(Path.Combine(workDir, "tunnel_url.txt"), foundUrl);
                    string docsDir = Path.Combine(workDir, "docs");
                    if (Directory.Exists(docsDir))
                    {
                        File.WriteAllText(Path.Combine(docsDir, "tunnel_url.json"), "{\"url\":\"" + foundUrl + "\"}");
                    }
                }
                catch { }
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
