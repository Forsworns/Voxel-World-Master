using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.Design;
using System.IO;
using System.Threading;


namespace CGGUI
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            this.backgroundWorker1.DoWork += backgroundWorker1_DoWork;//工作线程回调，将要执行的代码放在此函数里
            this.backgroundWorker1.ProgressChanged += backgroundWorker1_ProgressChanged;//当进度改变时回调
            this.backgroundWorker1.RunWorkerCompleted += new RunWorkerCompletedEventHandler(this.backgroundWorker1_RunWorkerCompleted);//当完成时回调
            this.backgroundWorker1.WorkerReportsProgress = true;//此属性必须设置，否则读取不到进度
            this.pwd = System.IO.Directory.GetCurrentDirectory();
        }


        //Stl2Obj
        private void button1_Click(object sender, EventArgs e)
        {
            
            if (MeshFileBox.Text.IndexOf(".stl") != -1)
            {
                Process myProcess = new Process();

                string exeName = this.pwd + @"\utils\stl2obj.exe";

                string para = MeshFileBox.Text + " " + objFilePath.Text + "\\" + objFileName.Text;

                ProcessStartInfo myProcessStartInfo = new ProcessStartInfo(exeName, para);

                myProcess.StartInfo = myProcessStartInfo;

                myProcess.Start();

                while (!myProcess.HasExited)

                {

                    myProcess.WaitForExit();

                }

                int returnValue = myProcess.ExitCode;
            }
            else if (MeshFileBox.Text.IndexOf(".obj") != -1)
            {
                DialogResult dr = MessageBox.Show("已经为OBJ文件，不需要此步转换！");
            }
            else
            {
                DialogResult dr = MessageBox.Show("请输入有效路径！");
            }
        }


        //初始文件选择
        private void button5_Click(object sender, EventArgs e)
        {
            this.objFileName.ResetText();
            this.objFilePath.ResetText();
            this.MeshFileBox.ResetText();
            this.VoxelPath.ResetText();
            this.pointCloudPath.ResetText();
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "Model(*.obj;*.stl;*.binvox;*.pcd)|*.obj;*.stl;*.binvox;*.pcd";
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                this.MeshFileBox.SelectedText = dialog.FileName;
                if (dialog.SafeFileName.IndexOf(".obj") != -1)
                {
                    this.objFileName.SelectedText = "已经为OBJ文件，不需要进行stl=>obj转换";
                    int num = dialog.FileName.IndexOf(dialog.SafeFileName);
                    this.objFilePath.SelectedText = dialog.FileName.Remove(num - 1);
                    this.VoxelPath.SelectedText = dialog.FileName.Remove(num - 1);
                    this.pointCloudPath.SelectedText = dialog.FileName.Remove(num - 1);
                }
                else if (dialog.SafeFileName.IndexOf(".stl") != -1)
                {
                    this.objFileName.SelectedText = dialog.SafeFileName.Replace(".stl", ".obj");
                    int num = dialog.FileName.IndexOf(dialog.SafeFileName);
                    this.objFilePath.SelectedText = dialog.FileName.Remove(num - 1);
                    this.VoxelPath.SelectedText = dialog.FileName.Remove(num - 1);
                    this.pointCloudPath.SelectedText = dialog.FileName.Remove(num - 1);
                }
                else {
                    this.objFileName.SelectedText = "选中用于重建的文件";
                    int num = dialog.FileName.IndexOf(dialog.SafeFileName);
                    this.objFilePath.SelectedText = dialog.FileName.Remove(num - 1);
                    this.VoxelPath.SelectedText = dialog.FileName.Remove(num - 1);
                    this.pointCloudPath.SelectedText = dialog.FileName.Remove(num - 1);
                }

            }
        }


        //Select stl2obj output path
        private void button6_Click(object sender, EventArgs e)
        {
            this.objFilePath.ResetText();
            System.Windows.Forms.FolderBrowserDialog folder = new System.Windows.Forms.FolderBrowserDialog();
            if (folder.ShowDialog() == DialogResult.OK)
            {
                this.objFilePath.Text = folder.SelectedPath;
            }
        }


        //体素化
        private void button3_Click(object sender, EventArgs e)
        {
            int size;
            int.TryParse(sizeBox.Text, out size);
            if (MeshFileBox.Text.IndexOf(".stl") != -1)
            {
                DialogResult dr = MessageBox.Show("请在最上方选择转换后的Obj文件！");
            }
            else if (MeshFileBox.Text.IndexOf(".obj") != -1)
            {
                if (sizeBox.Text.Length < 1 || size > 800)
                {
                    DialogResult dr = MessageBox.Show("请输入有效参数 size <= 800！");
                }
                else
                {
                    //体素化
                    Process myProcess = new Process();

                    string exeName = this.pwd + @"\utils\voxel_reconstruct.exe";                                                                               //voxel.exe 路径填写
                    string[] split = MeshFileBox.Text.Split(new char[] { '\\' });
                    string para = "-r" + ' ' + sizeBox.Text + ' ' + MeshFileBox.Text + ' ' + VoxelPath.Text + '\\' + split[split.Length - 1].Replace(".obj", "");

                    ProcessStartInfo myProcessStartInfo = new ProcessStartInfo(exeName, para);

                    myProcess.StartInfo = myProcessStartInfo;

                    myProcess.Start();

                    while (!myProcess.HasExited)

                    {

                        myProcess.WaitForExit();

                    }

                    int returnValue = myProcess.ExitCode;


                    //显示
                    voxelFile.ResetText();
                    voxelFile.Text = VoxelPath.Text + "\\" + split[split.Length - 1].Replace(".obj", ".binvox");
                    Process myProcess_show = new Process();

                    string exeName_show = this.pwd + @"\utils\viewvox.exe";                                                                               //voxelshow.exe 路径填写

                    string para_show = VoxelPath.Text + "\\" + split[split.Length - 1].Replace(".obj", ".binvox");

                    ProcessStartInfo myProcessStartInfo_show = new ProcessStartInfo(exeName_show, para_show);

                    myProcess_show.StartInfo = myProcessStartInfo_show;

                    myProcess_show.Start();

                    while (!myProcess_show.HasExited)

                    {

                        myProcess_show.WaitForExit();

                    }

                    int returnValue_show = myProcess_show.ExitCode;
                }
            }
            else
            {
                DialogResult dr = MessageBox.Show("请输入有效路径！");
            }
        }


        //Select Voxel Output Path
        private void button7_Click(object sender, EventArgs e)
        {
            this.VoxelPath.ResetText();
            System.Windows.Forms.FolderBrowserDialog folder = new System.Windows.Forms.FolderBrowserDialog();
            if (folder.ShowDialog() == DialogResult.OK)
            {
                this.VoxelPath.Text = folder.SelectedPath;
            }
        }


        //Select Voxel File Path
        private void button1_Click_1(object sender, EventArgs e)
        {
            this.voxelFile.ResetText();
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "Voxel(*.binvox;)|*.binvox;";
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                this.voxelFile.SelectedText = dialog.FileName;
            }
        }


        // Show Voxel
        private void ShowVoxel_Click(object sender, EventArgs e)
        {
            if (voxelFile.Text.IndexOf(".binvox") == -1)
            {
                DialogResult dr = MessageBox.Show("请输入有效路径！");
            }
            else
            {
                //显示
                Process myProcess_show = new Process();

                string exeName_show = this.pwd + @"\utils\viewvox.exe";                                                                               //voxelshow.exe 路径填写

                string para_show = voxelFile.Text;

                ProcessStartInfo myProcessStartInfo_show = new ProcessStartInfo(exeName_show, para_show);

                myProcess_show.StartInfo = myProcessStartInfo_show;

                myProcess_show.Start();

                while (!myProcess_show.HasExited)

                {

                    myProcess_show.WaitForExit();

                }

                int returnValue_show = myProcess_show.ExitCode;
            }
        }

        //Select Point Cloud Path
        private void button2_Click(object sender, EventArgs e)
        {
            this.pointCloudPath.ResetText();
            System.Windows.Forms.FolderBrowserDialog folder = new System.Windows.Forms.FolderBrowserDialog();
            if (folder.ShowDialog() == DialogResult.OK)
            {
                this.pointCloudPath.Text = folder.SelectedPath;
            }
        }


        //Point Cloud
        private void PointCloud_Click(object sender, EventArgs e)
        {
            if (MeshFileBox.Text.IndexOf(".stl") != -1)
            {
                DialogResult dr = MessageBox.Show("请在最上方选择转换后的Obj文件！");
            }
            else if (MeshFileBox.Text.IndexOf(".obj") != -1)
            {
                this.backgroundWorker1.RunWorkerAsync();
            }
            else
            {
                DialogResult dr = MessageBox.Show("请输入有效路径！");
            }
        }

        private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
        {
            backgroundWorker1.ReportProgress(0);
            FileInfo f = new FileInfo(MeshFileBox.Text);
            string[] split = MeshFileBox.Text.Split(new char[] { '\\' });
            FileStream fs = new FileStream(pointCloudPath.Text + '\\' + split[split.Length - 1].Replace(".obj", ".pcd"), FileMode.Create);
            StreamWriter sw = new StreamWriter(fs);
            sw.Write("VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH ");

            StreamReader sr = new StreamReader(MeshFileBox.Text, Encoding.Default);
            String line;
            int num = 0;
            String[] output = new string[100000000];
            while ((line = sr.ReadLine()) != null)
            {
                string[] v = line.Split(new char[] { ' ' });
                if (v[0] == "v")
                {
                    output[num] = (v[1] + ' ' + v[2] + ' ' + v[3] + '\n');
                    num++;
                }
            }
            sw.Write(num + "\nHEIGHT 1\nPOINTS " + num + "\nDATA ascii\n");
            for (int j = 0; j < num; ++j)
            {
                sw.Write(output[j]);
            }
            this.backgroundWorker1.ReportProgress(100);
            sw.Flush();
            sw.Close();
            fs.Close();
            return;
        }

        //进度条更新
        private void backgroundWorker1_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            this.progressBar1.Value = e.ProgressPercentage;
        }

        private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            MessageBox.Show("Convert to Point Cloud Completed!");
            this.progressBar1.Value = 100;
        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        // 选取重建的输出目录
        private void button3_Click_1(object sender, EventArgs e)
        {
            this.ReconstructPath.ResetText();
            System.Windows.Forms.FolderBrowserDialog folder = new System.Windows.Forms.FolderBrowserDialog();
            if (folder.ShowDialog() == DialogResult.OK)
            {
                this.ReconstructPath.Text = folder.SelectedPath;
            }
        }

        private void Reconstruction_Click(object sender, EventArgs e)
        {
            int fileType;
            string[] split = MeshFileBox.Text.Split(new char[] { '\\' });
            string fileName;
            if (MeshFileBox.Text.IndexOf(".binvox") != -1)
            {
                DialogResult dr = MessageBox.Show("处理体素文件");
                fileType = 0;
                fileName = fileName = split[split.Length - 1].Replace(".binvox", "");
            }
            else if (MeshFileBox.Text.IndexOf(".pcd") != -1)
            {
                DialogResult dr = MessageBox.Show("处理点云文件");
                fileType = 1;
                fileName = fileName = split[split.Length - 1].Replace(".pcd", "");
            }
            else
            {
                DialogResult dr = MessageBox.Show("请输入有效路径！");
                return ;
            }

            // 准备
            Process myProcess = new Process();
            
            string exeName = this.pwd + @"\utils\voxel_reconstruct.exe";                                                                               //reconstruct.exe 路径填写
            string para = "-f" + ' ' + '0' + ' ' + "-t" + ' ' + fileType + ' ' + MeshFileBox.Text + ' ' + ReconstructPath.Text + '\\' + fileName;

            ProcessStartInfo myProcessStartInfo = new ProcessStartInfo(exeName, para);

            myProcess.StartInfo = myProcessStartInfo;

            myProcess.Start();

            while (!myProcess.HasExited)

            {

                myProcess.WaitForExit();

            }
            int returnValue = myProcess.ExitCode;


            // 估计法向量参数
            myProcess = new Process();

            exeName = this.pwd + @"\utils\normal_estimate.exe";                                                                               
            para = this.pwd + @"\__to_cal_normal__.ply";

            myProcessStartInfo = new ProcessStartInfo(exeName, para);

            myProcess.StartInfo = myProcessStartInfo;

            myProcess.Start();

            while (!myProcess.HasExited)

            {

                myProcess.WaitForExit();

            }
            returnValue = myProcess.ExitCode;


            // 重构
            myProcess = new Process();

            exeName = this.pwd + @"\utils\voxel_reconstruct.exe";                                                                               //reconstruct.exe 路径填写
            para = "-f" + ' ' + '0' + ' ' + "-t" + ' ' + fileType + ' ' + "-p" + ' ' + '0' + ' ' + "-d" + ' ' + '0' + ' ' + MeshFileBox.Text + ' ' + ReconstructPath.Text + '\\' + fileName; // 再次运行

            myProcessStartInfo = new ProcessStartInfo(exeName, para);

            myProcess.StartInfo = myProcessStartInfo;

            myProcess.Start();

            while (!myProcess.HasExited)

            {

                myProcess.WaitForExit();

            }
            returnValue = myProcess.ExitCode;
            System.IO.File.Delete(this.pwd + @"\__normals__.txt");
            System.IO.File.Delete(this.pwd + @"\__to_cal_normal__.ply");
            DialogResult message = MessageBox.Show("mesh重构完毕");
        }
    }
}