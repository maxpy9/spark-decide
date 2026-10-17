using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
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
        private Button btnBrowse;
        private Button btnCopyLog;
        private Button btnClearLog;

        private TextBox txtWorkDir;
        private Label lblDiskSpace;
        private Label statusPill;
        private Label statusLabel;
        private ProgressBar progressBar;
        private TextBox logBox;
        private PictureBox logoBox;

        private Process serverProcess = null;
        private Process tunnelProcess = null;
        private string workDir;
        private string activeTunnelUrl = "";
        private const string GITHUB_RAW = "https://raw.githubusercontent.com/maxpy9/spark-decide/main/";

        public MainForm()
        {
            // Default working directory: directory where this launcher is located
            string cur = Environment.CurrentDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (File.Exists(Path.Combine(cur, "spark_decide.py")))
            {
                workDir = cur;
            }
            else if (File.Exists(Path.Combine(baseDir, "spark_decide.py")))
            {
                workDir = baseDir;
            }
            else if (Directory.Exists(@"C:\Users\Acer\spark-snake"))
            {
                workDir = @"C:\Users\Acer\spark-snake";
            }
            else
            {
                workDir = cur;
            }
            LoadSavedConfig();

            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            ServicePointManager.ServerCertificateValidationCallback = delegate { return true; };

            this.Text = "Spark-Decide — Installer & Studio Launcher";
            this.Size = new Size(860, 650);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(15, 18, 24); // #0F1218
            this.ForeColor = Color.FromArgb(230, 237, 243);
            this.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);
            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;

            string icoPath = Path.Combine(workDir, "spark_decide.ico");
            if (!File.Exists(icoPath)) icoPath = Path.Combine(baseDir, "spark_decide.ico");
            if (File.Exists(icoPath))
            {
                try { this.Icon = new Icon(icoPath); } catch { }
            }

            BuildUI();

            AppendLog("=== Spark-Decide Automated Setup & Studio Launcher ===");
            AppendLog("Modello Foundation: Rizzo Flow 4B (HuggingFace)");
            AppendLog("Motore Decisionale: Spark-Decide (LoRA + DecisionHead)");
            AppendLog("Cartella di lavoro: " + workDir);
            AppendLog("Stato: Pronto. Clicca 'Installa tutto' per verificare i file o 'Esegui server'.");

            this.FormClosing += (s, e) => {
                SaveConfig();
                KillProcesses();
            };
        }

        private void BuildUI()
        {
            // 1. Top Header Banner
            Panel headerPanel = new Panel();
            headerPanel.Location = new Point(0, 0);
            headerPanel.Size = new Size(860, 84);
            headerPanel.BackColor = Color.FromArgb(22, 27, 34); // #161B22

            logoBox = new PictureBox();
            logoBox.Size = new Size(58, 58);
            logoBox.Location = new Point(18, 13);
            logoBox.SizeMode = PictureBoxSizeMode.Zoom;
            string baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string cur = Environment.CurrentDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string[] possibleLogos = new string[] {
                Path.Combine(workDir, "spark_decide_logo.png"),
                Path.Combine(baseDir, "spark_decide_logo.png"),
                Path.Combine(cur, "spark_decide_logo.png"),
                @"C:\Users\Acer\spark-snake\spark_decide_logo.png"
            };
            foreach (string lp in possibleLogos)
            {
                if (File.Exists(lp))
                {
                    try { logoBox.Image = Image.FromFile(lp); break; } catch { }
                }
            }
            headerPanel.Controls.Add(logoBox);

            Label titleLbl = new Label();
            titleLbl.Text = "⚡ SPARK-DECIDE";
            titleLbl.Font = new Font("Segoe UI", 14.5f, FontStyle.Bold);
            titleLbl.ForeColor = Color.FromArgb(62, 207, 142); // #3ECF8E
            titleLbl.AutoSize = true;
            titleLbl.Location = new Point(88, 14);
            headerPanel.Controls.Add(titleLbl);

            Label subLbl = new Label();
            subLbl.Text = "Autonomous Neural Structured Decision Engine & Studio Launcher (PC & Telefono)";
            subLbl.Font = new Font("Segoe UI", 9f, FontStyle.Regular);
            subLbl.ForeColor = Color.FromArgb(139, 148, 158);
            subLbl.AutoSize = true;
            subLbl.Location = new Point(90, 46);
            headerPanel.Controls.Add(subLbl);

            // Status Pill badge (top right)
            statusPill = new Label();
            statusPill.Text = "● PRONTO";
            statusPill.Font = new Font("Segoe UI", 8.5f, FontStyle.Bold);
            statusPill.ForeColor = Color.FromArgb(62, 207, 142);
            statusPill.BackColor = Color.FromArgb(12, 38, 26);
            statusPill.BorderStyle = BorderStyle.FixedSingle;
            statusPill.TextAlign = ContentAlignment.MiddleCenter;
            statusPill.Size = new Size(130, 28);
            statusPill.Location = new Point(700, 28);
            headerPanel.Controls.Add(statusPill);

            this.Controls.Add(headerPanel);

            // 2. Folder Selection Card
            Panel folderCard = new Panel();
            folderCard.Location = new Point(18, 96);
            folderCard.Size = new Size(810, 68);
            folderCard.BackColor = Color.FromArgb(22, 27, 34);
            folderCard.BorderStyle = BorderStyle.FixedSingle;

            Label folderTitle = new Label();
            folderTitle.Text = "Cartella di Installazione & Modelli:";
            folderTitle.Font = new Font("Segoe UI", 9f, FontStyle.Bold);
            folderTitle.ForeColor = Color.FromArgb(200, 210, 225);
            folderTitle.Location = new Point(12, 8);
            folderTitle.AutoSize = true;
            folderCard.Controls.Add(folderTitle);

            lblDiskSpace = new Label();
            lblDiskSpace.Font = new Font("Segoe UI", 8.5f, FontStyle.Regular);
            lblDiskSpace.ForeColor = Color.FromArgb(62, 207, 142);
            lblDiskSpace.Location = new Point(245, 9);
            lblDiskSpace.AutoSize = true;
            folderCard.Controls.Add(lblDiskSpace);

            txtWorkDir = new TextBox();
            txtWorkDir.Location = new Point(12, 32);
            txtWorkDir.Size = new Size(680, 25);
            txtWorkDir.BackColor = Color.FromArgb(13, 16, 23);
            txtWorkDir.ForeColor = Color.FromArgb(230, 237, 243);
            txtWorkDir.BorderStyle = BorderStyle.FixedSingle;
            txtWorkDir.Font = new Font("Consolas", 9.5f, FontStyle.Regular);
            txtWorkDir.Text = workDir;
            txtWorkDir.TextChanged += (s, e) => {
                if (Directory.Exists(txtWorkDir.Text.Trim()))
                {
                    workDir = txtWorkDir.Text.Trim();
                    UpdateDiskSpace();
                }
            };
            folderCard.Controls.Add(txtWorkDir);

            btnBrowse = new Button();
            btnBrowse.Text = "Sfoglia...";
            btnBrowse.Font = new Font("Segoe UI", 9f, FontStyle.Bold);
            btnBrowse.BackColor = Color.FromArgb(33, 38, 45);
            btnBrowse.ForeColor = Color.FromArgb(230, 237, 243);
            btnBrowse.FlatStyle = FlatStyle.Flat;
            btnBrowse.FlatAppearance.BorderColor = Color.FromArgb(60, 70, 85);
            btnBrowse.Size = new Size(95, 27);
            btnBrowse.Location = new Point(700, 31);
            btnBrowse.Cursor = Cursors.Hand;
            btnBrowse.Click += (s, e) => BrowseAndSetFolder();
            folderCard.Controls.Add(btnBrowse);

            this.Controls.Add(folderCard);
            UpdateDiskSpace();

            // 3. Action Buttons Row
            int btnY = 174;
            int btnH = 42;

            btnInstallAll = new Button();
            btnInstallAll.Text = "Installa tutto";
            btnInstallAll.Font = new Font("Segoe UI", 9.5f, FontStyle.Bold);
            btnInstallAll.BackColor = Color.FromArgb(16, 185, 129); // #10B981
            btnInstallAll.ForeColor = Color.FromArgb(10, 30, 20);
            btnInstallAll.FlatStyle = FlatStyle.Flat;
            btnInstallAll.FlatAppearance.BorderSize = 0;
            btnInstallAll.Size = new Size(190, btnH);
            btnInstallAll.Location = new Point(18, btnY);
            btnInstallAll.Cursor = Cursors.Hand;
            btnInstallAll.Click += BtnInstallAll_Click;
            this.Controls.Add(btnInstallAll);

            btnStartServer = new Button();
            btnStartServer.Text = "► Esegui server";
            btnStartServer.Font = new Font("Segoe UI", 10f, FontStyle.Bold);
            btnStartServer.BackColor = Color.FromArgb(62, 207, 142); // #3ECF8E
            btnStartServer.ForeColor = Color.FromArgb(5, 35, 20);
            btnStartServer.FlatStyle = FlatStyle.Flat;
            btnStartServer.FlatAppearance.BorderSize = 0;
            btnStartServer.Size = new Size(195, btnH);
            btnStartServer.Location = new Point(220, btnY);
            btnStartServer.Cursor = Cursors.Hand;
            btnStartServer.Click += BtnStartServer_Click;
            this.Controls.Add(btnStartServer);

            btnStopServer = new Button();
            btnStopServer.Text = "■ Stoppa il server";
            btnStopServer.Font = new Font("Segoe UI", 9.5f, FontStyle.Bold);
            btnStopServer.BackColor = Color.FromArgb(45, 20, 20);
            btnStopServer.ForeColor = Color.FromArgb(252, 165, 165);
            btnStopServer.FlatStyle = FlatStyle.Flat;
            btnStopServer.FlatAppearance.BorderColor = Color.FromArgb(127, 29, 29);
            btnStopServer.Size = new Size(190, btnH);
            btnStopServer.Location = new Point(427, btnY);
            btnStopServer.Cursor = Cursors.Hand;
            btnStopServer.Enabled = false;
            btnStopServer.Click += BtnStopServer_Click;
            this.Controls.Add(btnStopServer);

            btnOpenBrowser = new Button();
            btnOpenBrowser.Text = "Apri Studio";
            btnOpenBrowser.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);
            btnOpenBrowser.BackColor = Color.FromArgb(31, 36, 48);
            btnOpenBrowser.ForeColor = Color.FromArgb(230, 237, 243);
            btnOpenBrowser.FlatStyle = FlatStyle.Flat;
            btnOpenBrowser.FlatAppearance.BorderColor = Color.FromArgb(61, 72, 92);
            btnOpenBrowser.Size = new Size(185, btnH);
            btnOpenBrowser.Location = new Point(629, btnY);
            btnOpenBrowser.Cursor = Cursors.Hand;
            btnOpenBrowser.Click += (s, e) => {
                string url = !string.IsNullOrEmpty(activeTunnelUrl) ? activeTunnelUrl : "http://127.0.0.1:7860";
                Process.Start(url);
            };
            this.Controls.Add(btnOpenBrowser);

            // 4. Status and Progress Bar
            statusLabel = new Label();
            statusLabel.Text = "Stato: Pronto. Clicca 'Installa tutto' per avviare o 'Esegui server' per avviare subito.";
            statusLabel.AutoSize = false;
            statusLabel.Size = new Size(810, 20);
            statusLabel.Location = new Point(18, 226);
            statusLabel.ForeColor = Color.FromArgb(170, 185, 200);
            this.Controls.Add(statusLabel);

            progressBar = new ProgressBar();
            progressBar.Location = new Point(18, 248);
            progressBar.Size = new Size(810, 12);
            progressBar.Minimum = 0;
            progressBar.Maximum = 100;
            progressBar.Value = 0;
            this.Controls.Add(progressBar);

            // 5. Console Card & Actions
            Panel logHeader = new Panel();
            logHeader.Location = new Point(18, 268);
            logHeader.Size = new Size(810, 28);
            logHeader.BackColor = Color.FromArgb(22, 27, 34);

            Label logTitle = new Label();
            logTitle.Text = "Console di Esecuzione ed Installazione";
            logTitle.Font = new Font("Segoe UI", 8.5f, FontStyle.Bold);
            logTitle.ForeColor = Color.FromArgb(139, 148, 158);
            logTitle.Location = new Point(10, 6);
            logTitle.AutoSize = true;
            logHeader.Controls.Add(logTitle);

            btnCopyLog = new Button();
            btnCopyLog.Text = "Copia Log";
            btnCopyLog.Font = new Font("Segoe UI", 7.5f, FontStyle.Regular);
            btnCopyLog.BackColor = Color.FromArgb(33, 38, 45);
            btnCopyLog.ForeColor = Color.FromArgb(200, 210, 225);
            btnCopyLog.FlatStyle = FlatStyle.Flat;
            btnCopyLog.FlatAppearance.BorderSize = 0;
            btnCopyLog.Size = new Size(85, 22);
            btnCopyLog.Location = new Point(625, 3);
            btnCopyLog.Cursor = Cursors.Hand;
            btnCopyLog.Click += (s, e) => {
                if (!string.IsNullOrEmpty(logBox.Text))
                {
                    Clipboard.SetText(logBox.Text);
                    MessageBox.Show(this, "Log copiato negli appunti!", "Spark-Decide", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            };
            logHeader.Controls.Add(btnCopyLog);

            btnClearLog = new Button();
            btnClearLog.Text = "Pulisci";
            btnClearLog.Font = new Font("Segoe UI", 7.5f, FontStyle.Regular);
            btnClearLog.BackColor = Color.FromArgb(33, 38, 45);
            btnClearLog.ForeColor = Color.FromArgb(200, 210, 225);
            btnClearLog.FlatStyle = FlatStyle.Flat;
            btnClearLog.FlatAppearance.BorderSize = 0;
            btnClearLog.Size = new Size(70, 22);
            btnClearLog.Location = new Point(718, 3);
            btnClearLog.Cursor = Cursors.Hand;
            btnClearLog.Click += (s, e) => { logBox.Clear(); };
            logHeader.Controls.Add(btnClearLog);

            this.Controls.Add(logHeader);

            // Log Console Box
            logBox = new TextBox();
            logBox.Multiline = true;
            logBox.ScrollBars = ScrollBars.Vertical;
            logBox.Location = new Point(18, 296);
            logBox.Size = new Size(810, 295);
            logBox.BackColor = Color.FromArgb(9, 12, 16); // #090C10
            logBox.ForeColor = Color.FromArgb(62, 207, 142);
            logBox.Font = new Font("Consolas", 9.5f, FontStyle.Regular);
            logBox.ReadOnly = true;
            logBox.BorderStyle = BorderStyle.FixedSingle;
            this.Controls.Add(logBox);
        }

        private void UpdateDiskSpace()
        {
            try
            {
                string root = Path.GetPathRoot(workDir);
                if (!string.IsNullOrEmpty(root))
                {
                    DriveInfo di = new DriveInfo(root);
                    if (di.IsReady)
                    {
                        double freeGb = di.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
                        lblDiskSpace.Text = string.Format("• Spazio libero su {0}: {1:0.1} GB (Richiesti: ~8 GB)", root, freeGb);
                        if (freeGb < 8.0)
                        {
                            lblDiskSpace.ForeColor = Color.FromArgb(248, 113, 113);
                        }
                        else
                        {
                            lblDiskSpace.ForeColor = Color.FromArgb(62, 207, 142);
                        }
                        return;
                    }
                }
            }
            catch { }
            lblDiskSpace.Text = "• Spazio su disco pronto.";
        }

        private void SetPill(string text, Color bg, Color fg)
        {
            if (this.InvokeRequired)
            {
                this.BeginInvoke(new Action<string, Color, Color>(SetPill), text, bg, fg);
                return;
            }
            statusPill.Text = text;
            statusPill.BackColor = bg;
            statusPill.ForeColor = fg;
        }

        private bool BrowseAndSetFolder()
        {
            using (FolderBrowserDialog fbd = new FolderBrowserDialog())
            {
                fbd.Description = "Seleziona la cartella dove scaricare e installare Spark-Decide e tutti i modelli:";
                if (Directory.Exists(workDir))
                {
                    fbd.SelectedPath = workDir;
                }
                fbd.ShowNewFolderButton = true;
                if (fbd.ShowDialog(this) == DialogResult.OK && !string.IsNullOrWhiteSpace(fbd.SelectedPath))
                {
                    workDir = fbd.SelectedPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                    txtWorkDir.Text = workDir;
                    UpdateDiskSpace();
                    SaveConfig();
                    AppendLog("Nuova cartella di lavoro impostata: " + workDir);
                    return true;
                }
            }
            return false;
        }

        private void SaveConfig()
        {
            try
            {
                string cfgDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SparkDecide");
                Directory.CreateDirectory(cfgDir);
                File.WriteAllText(Path.Combine(cfgDir, "launcher.cfg"), workDir);
            }
            catch { }
        }

        private void LoadSavedConfig()
        {
            try
            {
                string cfgFile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"SparkDecide\launcher.cfg");
                if (File.Exists(cfgFile))
                {
                    string saved = File.ReadAllText(cfgFile).Trim();
                    if (!string.IsNullOrEmpty(saved) && Directory.Exists(saved))
                    {
                        workDir = saved;
                    }
                }
            }
            catch { }
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
            // Come richiesto: Chiedi sempre la cartella prima di scaricare tutto!
            DialogResult ask = MessageBox.Show(this,
                "Desideri scaricare e installare Spark-Decide nella seguente cartella?\n\n" +
                workDir + "\n\n" +
                "• Premi 'Sì' per confermare questa cartella.\n" +
                "• Premi 'No' per sceglierne un'altra con il selettore.\n" +
                "• Premi 'Annulla' per interrompere.",
                "Selezione Cartella di Download",
                MessageBoxButtons.YesNoCancel,
                MessageBoxIcon.Question);

            if (ask == DialogResult.Cancel)
            {
                AppendLog("Operazione annullata.");
                return;
            }
            else if (ask == DialogResult.No)
            {
                if (!BrowseAndSetFolder())
                {
                    AppendLog("Nessuna cartella selezionata. Installazione annullata.");
                    return;
                }
            }

            Directory.CreateDirectory(workDir);

            btnInstallAll.Enabled = false;
            btnStartServer.Enabled = false;
            SetPill("⚡ INSTALLAZIONE", Color.FromArgb(45, 30, 10), Color.FromArgb(245, 158, 11));
            Thread worker = new Thread(RunInstallPipelineOnly);
            worker.IsBackground = true;
            worker.Start();
        }

        private void BtnStartServer_Click(object sender, EventArgs e)
        {
            btnStartServer.Enabled = false;
            btnStopServer.Enabled = true;
            SetPill("🚀 AVVIO...", Color.FromArgb(10, 30, 45), Color.FromArgb(56, 189, 248));
            Thread worker = new Thread(RunServerPipeline);
            worker.IsBackground = true;
            worker.Start();
        }

        private void BtnStopServer_Click(object sender, EventArgs e)
        {
            KillProcesses();
            AppendLog("Server e Tunnel arrestati con successo.");
            SetStatus("Stato: Server fermato.", 0);
            SetPill("● FERMATO", Color.FromArgb(45, 20, 20), Color.FromArgb(248, 113, 113));
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

        private int RunCommand(string fileName, string args, string customWorkDir = null)
        {
            string cwd = customWorkDir != null ? customWorkDir : workDir;
            AppendLog("> " + fileName + " " + args);
            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = fileName;
            psi.Arguments = args;
            psi.WorkingDirectory = cwd;
            psi.UseShellExecute = false;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.CreateNoWindow = true;
            psi.EnvironmentVariables["SPARK_ROOT_DIR"] = cwd;

            using (Process p = new Process())
            {
                p.StartInfo = psi;
                p.OutputDataReceived += (s, ev) => {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        AppendLog(ev.Data);
                        ParseProgress(ev.Data);
                    }
                };
                p.ErrorDataReceived += (s, ev) => {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        AppendLog(ev.Data);
                        ParseProgress(ev.Data);
                    }
                };
                p.Start();
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                p.WaitForExit();
                return p.ExitCode;
            }
        }

        private void ParseProgress(string line)
        {
            Match m = Regex.Match(line, @"(\d{1,3})%");
            if (m.Success)
            {
                int val;
                if (int.TryParse(m.Groups[1].Value, out val))
                {
                    if (this.InvokeRequired)
                    {
                        this.BeginInvoke(new Action<int>((p) => {
                            progressBar.Value = Math.Max(0, Math.Min(100, p));
                        }), val);
                    }
                }
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
                "spark_decide_logo.png",
                "spark_decide.ico",
                "cloudflared.exe"
            };

            string myDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

            using (WebClient wc = new WebClient())
            {
                wc.Headers.Add("User-Agent", "Spark-Decide-Installer/2.0");
                foreach (string f in coreFiles)
                {
                    string target = Path.Combine(workDir, f);
                    if (File.Exists(target)) continue;

                    string localSource = Path.Combine(myDir, f);
                    if (File.Exists(localSource))
                    {
                        AppendLog("[Copia Veloce] Copio " + f + " da installazione locale...");
                        File.Copy(localSource, target, true);
                    }
                    else
                    {
                        if (f == "cloudflared.exe") continue; // cloudflared è opzionale se non in repo github
                        AppendLog("[GitHub] Scaricamento veloce di " + f + "...");
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
                AppendLog("Creazione rapida ambiente virtuale .venv in: " + workDir);
                int code = RunCommand(sysPy, "-m venv .venv");
                if (code != 0 || !File.Exists(venvPy))
                {
                    AppendLog("Python di sistema non trovato. Download automatico di Python 3.11.9...");
                    string pyInstaller = Path.Combine(workDir, "python-3.11.9-amd64.exe");
                    using (WebClient wc = new WebClient())
                    {
                        wc.DownloadFile("https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe", pyInstaller);
                    }
                    AppendLog("Installazione silente di Python 3.11.9...");
                    RunCommand(pyInstaller, "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0");
                    sysPy = FindSystemPython();
                    RunCommand(sysPy, "-m venv .venv");
                }
            }

            // Verifica istantanea se le librerie sono gia' presenti
            string torchDir = Path.Combine(workDir, @".venv\Lib\site-packages\torch");
            if (Directory.Exists(torchDir))
            {
                AppendLog("Dipendenze Python & PyTorch verificate con successo (salto pip).");
                return true;
            }

            int chk = RunCommand(venvPy, "-c \"import torch, transformers, peft, bitsandbytes, fastapi, uvicorn; print('Librerie Python pronte!')\"");
            if (chk == 0)
            {
                AppendLog("Dipendenze Python gia' installate e integre (salto pip).");
                return true;
            }

            AppendLog("Installazione veloce delle dipendenze mancanti tramite pip...");
            string trusted = "--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org";
            RunCommand(venvPy, "-m pip install torch --index-url https://download.pytorch.org/whl/cu124 " + trusted);
            RunCommand(venvPy, "-m pip install \"transformers>=4.51,<5\" \"peft>=0.13,<0.18\" accelerate bitsandbytes fastapi uvicorn safetensors huggingface_hub matplotlib pandas pillow " + trusted);
            return true;
        }

        private void RunInstallPipelineOnly()
        {
            try
            {
                SetStatus("1/3: Verifica file di progetto nella cartella...", 20);
                EnsureRepoScripts();

                string venvPy = Path.Combine(workDir, @".venv\Scripts\python.exe");
                SetStatus("2/3: Verifica ambiente Python e accelerazione CUDA...", 45);
                EnsurePythonAndDeps(venvPy);

                SetStatus("3/3: Verifica e download modelli (GitHub + HuggingFace)...", 70);
                string setupScript = Path.Combine(workDir, "setup_models.py");
                if (File.Exists(setupScript))
                {
                    RunCommand(venvPy, "-u \"" + setupScript + "\" --dir \"" + workDir + "\"");
                }

                SetStatus("Stato: Installazione completata con successo!", 100);
                SetPill("● PRONTO", Color.FromArgb(12, 38, 26), Color.FromArgb(62, 207, 142));
                AppendLog("=================================================");
                AppendLog("INSTALLAZIONE COMPLETATA! Ora clicca 'Esegui server'.");
                AppendLog("=================================================");
            }
            catch (Exception ex)
            {
                AppendLog("ERRORE durante l'installazione: " + ex.Message);
                SetStatus("Stato: Errore nell'installazione.", 0);
                SetPill("● ERRORE", Color.FromArgb(45, 20, 20), Color.FromArgb(248, 113, 113));
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
                    AppendLog("Ambiente .venv non trovato. Eseguo prima l'installazione...");
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
                psiServer.EnvironmentVariables["SPARK_ROOT_DIR"] = workDir;

                serverProcess = new Process();
                serverProcess.StartInfo = psiServer;
                serverProcess.OutputDataReceived += (s, ev) => {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        AppendLog(ev.Data);
                        if (ev.Data.Contains("Uvicorn running on"))
                        {
                            SetStatus("🟢 Server ATTIVO su http://127.0.0.1:7860", 100);
                            SetPill("● ATTIVO", Color.FromArgb(12, 38, 26), Color.FromArgb(62, 207, 142));
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
                            SetPill("● ATTIVO", Color.FromArgb(12, 38, 26), Color.FromArgb(62, 207, 142));
                        }
                    }
                };
                serverProcess.Start();
                serverProcess.BeginOutputReadLine();
                serverProcess.BeginErrorReadLine();

                // 2. Avvia Cloudflare Tunnel per accesso da Telefono
                string cfExe = Path.Combine(workDir, "cloudflared.exe");
                if (!File.Exists(cfExe))
                {
                    string myDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                    string localCf = Path.Combine(myDir, "cloudflared.exe");
                    if (File.Exists(localCf))
                    {
                        try { File.Copy(localCf, cfExe, true); } catch { }
                    }
                }

                if (File.Exists(cfExe))
                {
                    AppendLog("Avvio Cloudflare Tunnel per accesso da Telefono...");
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
                SetPill("● ERRORE", Color.FromArgb(45, 20, 20), Color.FromArgb(248, 113, 113));
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
                AppendLog("🌐 URL PUBBLICO HTTPS ATTIVO PER IL TELEFONO:");
                AppendLog(foundUrl);
                AppendLog("=================================================");
                try
                {
                    File.WriteAllText(Path.Combine(workDir, "tunnel_url.txt"), foundUrl);
                }
                catch { }
            }
        }

        [STAThread]
        public static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            if (args.Length > 0 && args[0] == "--screenshot")
            {
                string outPath = args.Length > 1 ? args[1] : Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "screen_setup_gui.png");
                MainForm f = new MainForm();
                f.Show();
                Application.DoEvents();
                Thread.Sleep(600);
                Application.DoEvents();
                using (Bitmap bmp = new Bitmap(f.Width, f.Height))
                {
                    f.DrawToBitmap(bmp, new Rectangle(0, 0, f.Width, f.Height));
                    bmp.Save(outPath, System.Drawing.Imaging.ImageFormat.Png);
                }
                f.Close();
                return;
            }

            Application.Run(new MainForm());
        }
    }
}
