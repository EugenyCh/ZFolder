using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace IFForm
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private enum FractalType
        {
            Mand2D = 0,
            TMand2D = 1,
            Julia2D = 2,
            TJulia2D = 3,
            Mand3D = 4,
            TJulia3D = 5,
            Julia4D = 6,
            TJulia4D = 7,
            TMand4D = 8
        };

        private int WindowSize;
        private int FractalSize;
        private int Iterations;
        private int MaxFractalSize;
        private float TMand2DP;
        private float Julia2DCX;
        private float Julia2DCY;
        private float TJulia2DP;
        private float TJulia2DCX;
        private float TJulia2DCY;
        private float Mand3DP;
        private float TJulia3DP;
        private float TJulia3DCX;
        private float TJulia3DCY;
        private float TJulia3DCZ;
        private float QuatR;
        private float QuatA;
        private float QuatB;
        private float QuatC;

        private bool[] Flags = new bool[4];
        private bool MandFlag;
        private bool[] QuatFlags = new bool[4];
        private bool[] JuliaFlags = new bool[2];

        private ConsoleColor DefaultFore;
        public MainWindow()
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            DefaultFore = Console.ForegroundColor;
            InitializeComponent();
            ComboType.SelectedIndex = (int)(FractalType.Mand2D);
            BoxWinSize.Text = "500";
            BoxFractalSize.Text = "500";
            BoxIters.Text = "10";
            BoxMaxFractalSize.Text = "500";
            // TMand2D
            BoxTMand2DP.Text = "3.0";
            // Julia2D
            BoxJulia2DCX.Text = "-0.512511498387847";
            BoxJulia2DCY.Text = "0.521295573094847";
            // TJulia2D
            BoxTJulia2DP.Text = "3.0";
            BoxTJulia2DCX.Text = "-0.70176";
            BoxTJulia2DCY.Text = "-0.3842";
            // Mand3D
            BoxMand3DP.Text = "8.0";
            // TJulia3D
            BoxTJulia3DP.Text = "8.0";
            BoxTJulia3DCX.Text = "0.45";
            BoxTJulia3DCY.Text = "0.50";
            BoxTJulia3DCZ.Text = "0.55";
            // Julia4D
            BoxJulia4DQR.Text = "-0.65";
            BoxJulia4DQA.Text = "-0.5";
            BoxJulia4DQB.Text = "0.0";
            BoxJulia4DQC.Text = "0.0";
            // TJulia4D
            BoxTJulia4DP.Text = "4.0";
            BoxTJulia4DQR.Text = "-1.0";
            BoxTJulia4DQA.Text = "0.2";
            BoxTJulia4DQB.Text = "0.0";
            BoxTJulia4DQC.Text = "0.0";
            // TMand4D
            BoxTMand4DP.Text = "2.0";
            BoxTMand4DQC.Text = "0.0";
            Combo2D.SelectedIndex = 1;
            Combo3D.SelectedIndex = 1;
            Combo4D.SelectedIndex = 1;
        }

        private void ComboType_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // BlockD1 & BlockD2
            switch ((FractalType)ComboType.SelectedIndex)
            {
                case FractalType.Mand2D:
                case FractalType.TMand2D:
                case FractalType.Julia2D:
                case FractalType.TJulia2D:
                    BlockD1.Text = "^ 3";
                    break;
                case FractalType.Mand3D:
                case FractalType.TJulia3D:
                case FractalType.Julia4D:
                case FractalType.TJulia4D:
                case FractalType.TMand4D:
                    BlockD1.Text = "^ 3";
                    break;
            }
            BlockD2.Text = BlockD1.Text;
            // Combo Gradient
            switch ((FractalType)ComboType.SelectedIndex)
            {
                case FractalType.Mand2D:
                case FractalType.TMand2D:
                case FractalType.Julia2D:
                case FractalType.TJulia2D:
                    Combo2D.Visibility = Visibility.Visible;
                    Combo3D.Visibility = Visibility.Collapsed;
                    Combo4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.Mand3D:
                case FractalType.TJulia3D:
                    Combo2D.Visibility = Visibility.Collapsed;
                    Combo3D.Visibility = Visibility.Visible;
                    Combo4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.Julia4D:
                case FractalType.TJulia4D:
                case FractalType.TMand4D:
                    Combo2D.Visibility = Visibility.Collapsed;
                    Combo3D.Visibility = Visibility.Collapsed;
                    Combo4D.Visibility = Visibility.Visible;
                    break;
            }
            // Panel
            switch ((FractalType)ComboType.SelectedIndex)
            {
                case FractalType.Mand2D:
                    PanelTMand2D.Visibility = Visibility.Collapsed;
                    PanelJulia2D.Visibility = Visibility.Collapsed;
                    PanelTJulia2D.Visibility = Visibility.Collapsed;
                    PanelMand3D.Visibility = Visibility.Collapsed;
                    PanelTJulia3D.Visibility = Visibility.Collapsed;
                    PanelJulia4D.Visibility = Visibility.Collapsed;
                    PanelTJulia4D.Visibility = Visibility.Collapsed;
                    PanelTMand4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.TMand2D:
                    PanelTMand2D.Visibility = Visibility.Visible;
                    PanelJulia2D.Visibility = Visibility.Collapsed;
                    PanelTJulia2D.Visibility = Visibility.Collapsed;
                    PanelMand3D.Visibility = Visibility.Collapsed;
                    PanelTJulia3D.Visibility = Visibility.Collapsed;
                    PanelJulia4D.Visibility = Visibility.Collapsed;
                    PanelTJulia4D.Visibility = Visibility.Collapsed;
                    PanelTMand4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.Julia2D:
                    PanelTMand2D.Visibility = Visibility.Collapsed;
                    PanelJulia2D.Visibility = Visibility.Visible;
                    PanelTJulia2D.Visibility = Visibility.Collapsed;
                    PanelMand3D.Visibility = Visibility.Collapsed;
                    PanelTJulia3D.Visibility = Visibility.Collapsed;
                    PanelJulia4D.Visibility = Visibility.Collapsed;
                    PanelTJulia4D.Visibility = Visibility.Collapsed;
                    PanelTMand4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.TJulia2D:
                    PanelTMand2D.Visibility = Visibility.Collapsed;
                    PanelJulia2D.Visibility = Visibility.Collapsed;
                    PanelTJulia2D.Visibility = Visibility.Visible;
                    PanelMand3D.Visibility = Visibility.Collapsed;
                    PanelTJulia3D.Visibility = Visibility.Collapsed;
                    PanelJulia4D.Visibility = Visibility.Collapsed;
                    PanelTJulia4D.Visibility = Visibility.Collapsed;
                    PanelTMand4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.Mand3D:
                    PanelTMand2D.Visibility = Visibility.Collapsed;
                    PanelJulia2D.Visibility = Visibility.Collapsed;
                    PanelTJulia2D.Visibility = Visibility.Collapsed;
                    PanelMand3D.Visibility = Visibility.Visible;
                    PanelTJulia3D.Visibility = Visibility.Collapsed;
                    PanelJulia4D.Visibility = Visibility.Collapsed;
                    PanelTJulia4D.Visibility = Visibility.Collapsed;
                    PanelTMand4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.TJulia3D:
                    PanelTMand2D.Visibility = Visibility.Collapsed;
                    PanelJulia2D.Visibility = Visibility.Collapsed;
                    PanelTJulia2D.Visibility = Visibility.Collapsed;
                    PanelMand3D.Visibility = Visibility.Collapsed;
                    PanelTJulia3D.Visibility = Visibility.Visible;
                    PanelJulia4D.Visibility = Visibility.Collapsed;
                    PanelTJulia4D.Visibility = Visibility.Collapsed;
                    PanelTMand4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.Julia4D:
                    PanelTMand2D.Visibility = Visibility.Collapsed;
                    PanelJulia2D.Visibility = Visibility.Collapsed;
                    PanelTJulia2D.Visibility = Visibility.Collapsed;
                    PanelMand3D.Visibility = Visibility.Collapsed;
                    PanelTJulia3D.Visibility = Visibility.Collapsed;
                    PanelJulia4D.Visibility = Visibility.Visible;
                    PanelTJulia4D.Visibility = Visibility.Collapsed;
                    PanelTMand4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.TJulia4D:
                    PanelTMand2D.Visibility = Visibility.Collapsed;
                    PanelJulia2D.Visibility = Visibility.Collapsed;
                    PanelTJulia2D.Visibility = Visibility.Collapsed;
                    PanelMand3D.Visibility = Visibility.Collapsed;
                    PanelTJulia3D.Visibility = Visibility.Collapsed;
                    PanelJulia4D.Visibility = Visibility.Collapsed;
                    PanelTJulia4D.Visibility = Visibility.Visible;
                    PanelTMand4D.Visibility = Visibility.Collapsed;
                    break;
                case FractalType.TMand4D:
                    PanelTMand2D.Visibility = Visibility.Collapsed;
                    PanelJulia2D.Visibility = Visibility.Collapsed;
                    PanelTJulia2D.Visibility = Visibility.Collapsed;
                    PanelMand3D.Visibility = Visibility.Collapsed;
                    PanelTJulia3D.Visibility = Visibility.Collapsed;
                    PanelJulia4D.Visibility = Visibility.Collapsed;
                    PanelTJulia4D.Visibility = Visibility.Collapsed;
                    PanelTMand4D.Visibility = Visibility.Visible;
                    break;
            }
        }

        #region Left Panel
        private void BoxWinSize_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                WindowSize = int.Parse(BoxWinSize.Text);
                if (WindowSize < 100)
                    throw new Exception();
                BoxWinSize.Background = new SolidColorBrush(Colors.White);
                Flags[0] = true;
            }
            catch
            {
                BoxWinSize.Background = new SolidColorBrush(Colors.Pink);
                Flags[0] = false;
            }
        }

        private void BoxFractalSize_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                FractalSize = int.Parse(BoxFractalSize.Text);
                if (FractalSize < 100)
                    throw new Exception();
                BoxFractalSize.Background = new SolidColorBrush(Colors.White);
                Flags[1] = true;
            }
            catch
            {
                BoxFractalSize.Background = new SolidColorBrush(Colors.Pink);
                Flags[1] = false;
            }
        }

        private void BoxIters_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Iterations = int.Parse(BoxIters.Text);
                if (Iterations < 0)
                    throw new Exception();
                BoxIters.Background = new SolidColorBrush(Colors.White);
                Flags[2] = true;
            }
            catch
            {
                BoxIters.Background = new SolidColorBrush(Colors.Pink);
                Flags[2] = false;
            }
        }

        private void BoxMaxFractalSize_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                MaxFractalSize = int.Parse(BoxMaxFractalSize.Text);
                if (MaxFractalSize < 100)
                    throw new Exception();
                BoxMaxFractalSize.Background = new SolidColorBrush(Colors.White);
                Flags[3] = true;
            }
            catch
            {
                BoxMaxFractalSize.Background = new SolidColorBrush(Colors.Pink);
                Flags[3] = false;
            }
        }
        #endregion

        #region TMand2D
        private void BoxTMand2DP_TextChanged(object sender, TextChangedEventArgs e)
        {

        }
        #endregion

        #region Julia2D
        private void BoxJulia2DCX_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Julia2DCX = float.Parse(BoxJulia2DCX.Text);
                BoxJulia2DCX.Background = new SolidColorBrush(Colors.White);
                JuliaFlags[0] = true;
            }
            catch
            {
                BoxJulia2DCX.Background = new SolidColorBrush(Colors.Pink);
                JuliaFlags[0] = false;
            }
        }

        private void BoxJulia2DCY_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Julia2DCY = float.Parse(BoxJulia2DCY.Text);
                BoxJulia2DCY.Background = new SolidColorBrush(Colors.White);
                JuliaFlags[1] = true;
            }
            catch
            {
                BoxJulia2DCY.Background = new SolidColorBrush(Colors.Pink);
                JuliaFlags[1] = false;
            }
        }
        #endregion

        #region TJulia2D
        private void BoxTJulia2DCX_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTJulia2DCY_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTJulia2DP_TextChanged(object sender, TextChangedEventArgs e)
        {

        }
        #endregion

        #region Mand3D
        private void BoxMand3DP_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Mand3DP = float.Parse(BoxMand3DP.Text);
                if (Mand3DP < 2.0)
                    throw new Exception();
                BoxMand3DP.Background = new SolidColorBrush(Colors.White);
                MandFlag = true;
            }
            catch
            {
                BoxMand3DP.Background = new SolidColorBrush(Colors.Pink);
                MandFlag = false;
            }
        }
        #endregion

        #region TJulia3D
        private void BoxTJulia3DCX_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTJulia3DCY_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTJulia3DCZ_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTJulia3DP_TextChanged(object sender, TextChangedEventArgs e)
        {

        }
        #endregion

        #region Julia4D
        private void BoxJulia4DQR_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                QuatR = float.Parse(BoxJulia4DQR.Text);
                BoxJulia4DQR.Background = new SolidColorBrush(Colors.White);
                QuatFlags[0] = true;
            }
            catch
            {
                BoxJulia4DQR.Background = new SolidColorBrush(Colors.Pink);
                QuatFlags[0] = false;
            }
        }

        private void BoxJulia4DQA_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                QuatA = float.Parse(BoxJulia4DQA.Text);
                BoxJulia4DQA.Background = new SolidColorBrush(Colors.White);
                QuatFlags[1] = true;
            }
            catch
            {
                BoxJulia4DQA.Background = new SolidColorBrush(Colors.Pink);
                QuatFlags[1] = false;
            }
        }

        private void BoxJulia4DQB_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                QuatB = float.Parse(BoxJulia4DQB.Text);
                BoxJulia4DQB.Background = new SolidColorBrush(Colors.White);
                QuatFlags[2] = true;
            }
            catch
            {
                BoxJulia4DQB.Background = new SolidColorBrush(Colors.Pink);
                QuatFlags[2] = false;
            }
        }

        private void BoxJulia4DQC_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                QuatC = float.Parse(BoxJulia4DQC.Text);
                BoxJulia4DQC.Background = new SolidColorBrush(Colors.White);
                QuatFlags[3] = true;
            }
            catch
            {
                BoxJulia4DQC.Background = new SolidColorBrush(Colors.Pink);
                QuatFlags[3] = false;
            }
        }
        #endregion

        #region TJulia4D
        private void BoxTJulia4DQR_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTJulia4DQA_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTJulia4DQB_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTJulia4DQC_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTJulia4DP_TextChanged(object sender, TextChangedEventArgs e)
        {

        }
        #endregion

        #region TMand4D
        private void BoxTMand4DQC_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void BoxTMand4DP_TextChanged(object sender, TextChangedEventArgs e)
        {

        }
        #endregion

        private bool ValidationTest()
        {
            if (Flags[0] == false)
            {
                MessageBox.Show("Ошибка при заполнении размера окна: должно быть целое число >= 100",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (Flags[1] == false)
            {
                MessageBox.Show("Ошибка при заполнении размера фрактала: должно быть целое число >= 100",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (Flags[2] == false)
            {
                MessageBox.Show("Ошибка при заполнении количества итераций: должно быть целое число >= 0",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (Flags[3] == false)
            {
                MessageBox.Show("Ошибка при заполнении максимального размера фрактала: должно быть целое число >= 100",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (MandFlag == false)
            {
                MessageBox.Show("Ошибка при заполнении степени оболочки Мандельброта: должно быть число >= 2.0",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (QuatFlags[0] == false)
            {
                MessageBox.Show("Ошибка при заполнении компоненты R: должно быть число",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (QuatFlags[1] == false)
            {
                MessageBox.Show("Ошибка при заполнении компоненты A: должно быть число",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (QuatFlags[2] == false)
            {
                MessageBox.Show("Ошибка при заполнении компоненты B: должно быть число",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (QuatFlags[3] == false)
            {
                MessageBox.Show("Ошибка при заполнении компоненты C: должно быть число",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (JuliaFlags[0] == false)
            {
                MessageBox.Show("Ошибка при заполнении компоненты X: должно быть число",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (JuliaFlags[1] == false)
            {
                MessageBox.Show("Ошибка при заполнении компоненты Y: должно быть число",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            return true;
        }

        private void WriteInt(BinaryWriter writer, int value)
        {
            Console.Write($"{sizeof(int)} bytes : ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(value);
            Console.ForegroundColor = DefaultFore;
            writer.Write(value);
        }

        private void WriteFloat(BinaryWriter writer, float value)
        {
            Console.Write($"{sizeof(float)} bytes : ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(value);
            Console.ForegroundColor = DefaultFore;
            writer.Write(value);
        }

        private void ButtonRender_Click(object sender, RoutedEventArgs e)
        {
            if (!ValidationTest())
                return;
            try
            {
                using (BinaryWriter writer = new BinaryWriter(File.Open("input.bin", FileMode.Create)))
                {
                    Console.WriteLine("Generation of \"input.bin\"...");
                    WriteInt(writer, WindowSize);
                    WriteInt(writer, FractalSize);
                    WriteInt(writer, Iterations);
                    WriteInt(writer, MaxFractalSize);
                    switch ((FractalType)ComboType.SelectedIndex)
                    {
                        case FractalType.Mand2D:
                            WriteInt(writer, Combo2D.SelectedIndex);
                            Console.WriteLine("Type: Mandelbrot 2D");
                            break;
                        case FractalType.Julia2D:
                            WriteInt(writer, Combo2D.SelectedIndex);
                            WriteFloat(writer, Julia2DCX);
                            WriteFloat(writer, Julia2DCY);
                            Console.WriteLine("Type: Julia 2D");
                            break;
                        case FractalType.Mand3D:
                            WriteInt(writer, Combo3D.SelectedIndex);
                            WriteFloat(writer, Mand3DP);
                            Console.WriteLine("Type: Mandelbulb 3D");
                            break;
                        case FractalType.Julia4D:
                            WriteInt(writer, Combo4D.SelectedIndex);
                            WriteFloat(writer, QuatR);
                            WriteFloat(writer, QuatA);
                            WriteFloat(writer, QuatB);
                            WriteFloat(writer, QuatC);
                            WriteInt(writer, ComboJulia4DComponent.SelectedIndex);
                            Console.WriteLine("Type: Julia 4D");
                            break;
                    }
                    Console.WriteLine($"Total {writer.BaseStream.Length} bytes.");
                }
            }
            catch
            {
                MessageBox.Show("Ошибка при создании промежуточного файла",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            try
            {
                Process process;
                switch ((FractalType)ComboType.SelectedIndex)
                {
                    case FractalType.Mand2D:
                        process = new Process
                        {
                            StartInfo = new ProcessStartInfo
                            {
                                FileName = "DeviceMandelbrot2D.exe",
                                UseShellExecute = false,
                                CreateNoWindow = true
                            }
                        };
                        process.Start();
                        break;
                    case FractalType.Julia2D:
                        process = new Process
                        {
                            StartInfo = new ProcessStartInfo
                            {
                                FileName = "DeviceJulia2D.exe",
                                UseShellExecute = false,
                                CreateNoWindow = true
                            }
                        };
                        process.Start();
                        break;
                    case FractalType.Mand3D:
                        process = new Process
                        {
                            StartInfo = new ProcessStartInfo
                            {
                                FileName = "DeviceMandelbulb3D.exe",
                                UseShellExecute = false,
                                CreateNoWindow = true
                            }
                        };
                        process.Start();
                        break;
                    case FractalType.Julia4D:
                        process = new Process
                        {
                            StartInfo = new ProcessStartInfo
                            {
                                FileName = "DeviceJulia4D.exe",
                                UseShellExecute = false,
                                CreateNoWindow = true
                            }
                        };
                        process.Start();
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}
