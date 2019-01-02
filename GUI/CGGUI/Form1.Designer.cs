namespace CGGUI
{
    partial class Form1
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.STL2OBJ = new System.Windows.Forms.Button();
            this.PointCloud = new System.Windows.Forms.Button();
            this.Voxel = new System.Windows.Forms.Button();
            this.Reconstruction = new System.Windows.Forms.Button();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.button5 = new System.Windows.Forms.Button();
            this.MeshFileBox = new System.Windows.Forms.TextBox();
            this.objFilePath = new System.Windows.Forms.TextBox();
            this.button6 = new System.Windows.Forms.Button();
            this.objFileName = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.VoxelPath = new System.Windows.Forms.TextBox();
            this.button7 = new System.Windows.Forms.Button();
            this.sizeBox = new System.Windows.Forms.TextBox();
            this.size = new System.Windows.Forms.Label();
            this.ShowVoxel = new System.Windows.Forms.Button();
            this.voxelFile = new System.Windows.Forms.TextBox();
            this.button1 = new System.Windows.Forms.Button();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.pointCloudPath = new System.Windows.Forms.TextBox();
            this.button2 = new System.Windows.Forms.Button();
            this.backgroundWorker1 = new System.ComponentModel.BackgroundWorker();
            this.progressBar1 = new System.Windows.Forms.ProgressBar();
            this.ReconstructPath = new System.Windows.Forms.TextBox();
            this.button3 = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.SuspendLayout();
            // 
            // STL2OBJ
            // 
            this.STL2OBJ.Location = new System.Drawing.Point(101, 190);
            this.STL2OBJ.Margin = new System.Windows.Forms.Padding(4);
            this.STL2OBJ.Name = "STL2OBJ";
            this.STL2OBJ.Size = new System.Drawing.Size(137, 48);
            this.STL2OBJ.TabIndex = 0;
            this.STL2OBJ.Text = "STL2OBJ";
            this.STL2OBJ.UseVisualStyleBackColor = true;
            this.STL2OBJ.Click += new System.EventHandler(this.button1_Click);
            // 
            // PointCloud
            // 
            this.PointCloud.Location = new System.Drawing.Point(101, 270);
            this.PointCloud.Margin = new System.Windows.Forms.Padding(4);
            this.PointCloud.Name = "PointCloud";
            this.PointCloud.Size = new System.Drawing.Size(137, 46);
            this.PointCloud.TabIndex = 1;
            this.PointCloud.Text = "PointCloud";
            this.PointCloud.UseVisualStyleBackColor = true;
            this.PointCloud.Click += new System.EventHandler(this.PointCloud_Click);
            // 
            // Voxel
            // 
            this.Voxel.Location = new System.Drawing.Point(101, 348);
            this.Voxel.Margin = new System.Windows.Forms.Padding(4);
            this.Voxel.Name = "Voxel";
            this.Voxel.Size = new System.Drawing.Size(137, 46);
            this.Voxel.TabIndex = 2;
            this.Voxel.Text = "Voxel Grid";
            this.Voxel.UseVisualStyleBackColor = true;
            this.Voxel.Click += new System.EventHandler(this.button3_Click);
            // 
            // Reconstruction
            // 
            this.Reconstruction.Location = new System.Drawing.Point(101, 504);
            this.Reconstruction.Margin = new System.Windows.Forms.Padding(4);
            this.Reconstruction.Name = "Reconstruction";
            this.Reconstruction.Size = new System.Drawing.Size(137, 44);
            this.Reconstruction.TabIndex = 3;
            this.Reconstruction.Text = "Mesh Reconstruction";
            this.Reconstruction.UseVisualStyleBackColor = true;
            this.Reconstruction.Click += new System.EventHandler(this.Reconstruction_Click);
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // button5
            // 
            this.button5.Location = new System.Drawing.Point(877, 128);
            this.button5.Margin = new System.Windows.Forms.Padding(4);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(139, 26);
            this.button5.TabIndex = 6;
            this.button5.Text = "浏览文件";
            this.button5.UseVisualStyleBackColor = true;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // MeshFileBox
            // 
            this.MeshFileBox.Location = new System.Drawing.Point(101, 128);
            this.MeshFileBox.Margin = new System.Windows.Forms.Padding(4);
            this.MeshFileBox.Name = "MeshFileBox";
            this.MeshFileBox.Size = new System.Drawing.Size(767, 25);
            this.MeshFileBox.TabIndex = 7;
            // 
            // objFilePath
            // 
            this.objFilePath.Location = new System.Drawing.Point(247, 202);
            this.objFilePath.Margin = new System.Windows.Forms.Padding(4);
            this.objFilePath.Name = "objFilePath";
            this.objFilePath.Size = new System.Drawing.Size(223, 25);
            this.objFilePath.TabIndex = 8;
            // 
            // button6
            // 
            this.button6.Location = new System.Drawing.Point(479, 202);
            this.button6.Margin = new System.Windows.Forms.Padding(4);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(139, 26);
            this.button6.TabIndex = 9;
            this.button6.Text = "输出路径";
            this.button6.UseVisualStyleBackColor = true;
            this.button6.Click += new System.EventHandler(this.button6_Click);
            // 
            // objFileName
            // 
            this.objFileName.Location = new System.Drawing.Point(625, 202);
            this.objFileName.Margin = new System.Windows.Forms.Padding(4);
            this.objFileName.Name = "objFileName";
            this.objFileName.Size = new System.Drawing.Size(243, 25);
            this.objFileName.TabIndex = 10;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(900, 208);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(82, 15);
            this.label1.TabIndex = 11;
            this.label1.Text = "输出文件名";
            // 
            // VoxelPath
            // 
            this.VoxelPath.Location = new System.Drawing.Point(247, 359);
            this.VoxelPath.Margin = new System.Windows.Forms.Padding(4);
            this.VoxelPath.Name = "VoxelPath";
            this.VoxelPath.Size = new System.Drawing.Size(369, 25);
            this.VoxelPath.TabIndex = 12;
            // 
            // button7
            // 
            this.button7.Location = new System.Drawing.Point(625, 359);
            this.button7.Margin = new System.Windows.Forms.Padding(4);
            this.button7.Name = "button7";
            this.button7.Size = new System.Drawing.Size(139, 26);
            this.button7.TabIndex = 13;
            this.button7.Text = "输出路径";
            this.button7.UseVisualStyleBackColor = true;
            this.button7.Click += new System.EventHandler(this.button7_Click);
            // 
            // sizeBox
            // 
            this.sizeBox.Location = new System.Drawing.Point(768, 359);
            this.sizeBox.Margin = new System.Windows.Forms.Padding(4);
            this.sizeBox.Name = "sizeBox";
            this.sizeBox.Size = new System.Drawing.Size(100, 25);
            this.sizeBox.TabIndex = 14;
            // 
            // size
            // 
            this.size.AutoSize = true;
            this.size.Location = new System.Drawing.Point(900, 364);
            this.size.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.size.Name = "size";
            this.size.Size = new System.Drawing.Size(52, 15);
            this.size.TabIndex = 15;
            this.size.Text = "分辨率";
            // 
            // ShowVoxel
            // 
            this.ShowVoxel.Location = new System.Drawing.Point(101, 424);
            this.ShowVoxel.Margin = new System.Windows.Forms.Padding(4);
            this.ShowVoxel.Name = "ShowVoxel";
            this.ShowVoxel.Size = new System.Drawing.Size(137, 46);
            this.ShowVoxel.TabIndex = 16;
            this.ShowVoxel.Text = "ShowVoxel";
            this.ShowVoxel.UseVisualStyleBackColor = true;
            this.ShowVoxel.Click += new System.EventHandler(this.ShowVoxel_Click);
            // 
            // voxelFile
            // 
            this.voxelFile.Location = new System.Drawing.Point(247, 435);
            this.voxelFile.Margin = new System.Windows.Forms.Padding(4);
            this.voxelFile.Name = "voxelFile";
            this.voxelFile.Size = new System.Drawing.Size(621, 25);
            this.voxelFile.TabIndex = 17;
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(877, 435);
            this.button1.Margin = new System.Windows.Forms.Padding(4);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(139, 26);
            this.button1.TabIndex = 18;
            this.button1.Text = "浏览文件";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click_1);
            // 
            // pictureBox1
            // 
            this.pictureBox1.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("pictureBox1.BackgroundImage")));
            this.pictureBox1.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.pictureBox1.InitialImage = null;
            this.pictureBox1.Location = new System.Drawing.Point(101, 59);
            this.pictureBox1.Margin = new System.Windows.Forms.Padding(4);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(915, 50);
            this.pictureBox1.TabIndex = 19;
            this.pictureBox1.TabStop = false;
            // 
            // pointCloudPath
            // 
            this.pointCloudPath.Location = new System.Drawing.Point(247, 281);
            this.pointCloudPath.Margin = new System.Windows.Forms.Padding(4);
            this.pointCloudPath.Name = "pointCloudPath";
            this.pointCloudPath.Size = new System.Drawing.Size(621, 25);
            this.pointCloudPath.TabIndex = 20;
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(877, 282);
            this.button2.Margin = new System.Windows.Forms.Padding(4);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(139, 25);
            this.button2.TabIndex = 21;
            this.button2.Text = "输出路径";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // backgroundWorker1
            // 
            this.backgroundWorker1.WorkerReportsProgress = true;
            this.backgroundWorker1.DoWork += new System.ComponentModel.DoWorkEventHandler(this.backgroundWorker1_DoWork);
            // 
            // progressBar1
            // 
            this.progressBar1.Cursor = System.Windows.Forms.Cursors.AppStarting;
            this.progressBar1.Location = new System.Drawing.Point(247, 304);
            this.progressBar1.Margin = new System.Windows.Forms.Padding(4);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(623, 12);
            this.progressBar1.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.progressBar1.TabIndex = 22;
            // 
            // ReconstructPath
            // 
            this.ReconstructPath.Location = new System.Drawing.Point(249, 516);
            this.ReconstructPath.Margin = new System.Windows.Forms.Padding(4);
            this.ReconstructPath.Name = "ReconstructPath";
            this.ReconstructPath.Size = new System.Drawing.Size(619, 25);
            this.ReconstructPath.TabIndex = 23;
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(877, 516);
            this.button3.Margin = new System.Windows.Forms.Padding(4);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(139, 26);
            this.button3.TabIndex = 24;
            this.button3.Text = "输出路径";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click_1);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1067, 562);
            this.Controls.Add(this.button3);
            this.Controls.Add(this.ReconstructPath);
            this.Controls.Add(this.progressBar1);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.pointCloudPath);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.voxelFile);
            this.Controls.Add(this.ShowVoxel);
            this.Controls.Add(this.size);
            this.Controls.Add(this.sizeBox);
            this.Controls.Add(this.button7);
            this.Controls.Add(this.VoxelPath);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.objFileName);
            this.Controls.Add(this.button6);
            this.Controls.Add(this.objFilePath);
            this.Controls.Add(this.MeshFileBox);
            this.Controls.Add(this.button5);
            this.Controls.Add(this.Reconstruction);
            this.Controls.Add(this.Voxel);
            this.Controls.Add(this.PointCloud);
            this.Controls.Add(this.STL2OBJ);
            this.Margin = new System.Windows.Forms.Padding(4);
            this.Name = "Form1";
            this.Text = "AG";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button STL2OBJ;
        private System.Windows.Forms.Button PointCloud;
        private System.Windows.Forms.Button Voxel;
        private System.Windows.Forms.Button Reconstruction;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.TextBox MeshFileBox;
        private System.Windows.Forms.TextBox objFilePath;
        private System.Windows.Forms.Button button6;
        private System.Windows.Forms.TextBox objFileName;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox VoxelPath;
        private System.Windows.Forms.Button button7;
        private System.Windows.Forms.TextBox sizeBox;
        private System.Windows.Forms.Label size;
        private System.Windows.Forms.Button ShowVoxel;
        private System.Windows.Forms.TextBox voxelFile;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.TextBox pointCloudPath;
        private System.Windows.Forms.Button button2;
        private System.ComponentModel.BackgroundWorker backgroundWorker1;
        private System.Windows.Forms.ProgressBar progressBar1;
        private System.Windows.Forms.TextBox ReconstructPath;
        private System.Windows.Forms.Button button3;
        private string pwd;
    }
}

