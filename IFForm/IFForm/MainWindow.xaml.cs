using System;
using System.Collections.Generic;
using System.Globalization;
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
            Julia2D = 1,
            Mand3D = 2,
            Julia4D = 3
        };

        private int WindowSize;
        private int FractalSize;
        private int Iterations;
        private int MaxFractalSize;
        private float MandP;
        private float QuatR;
        private float QuatA;
        private float QuatB;
        private float QuatC;
        private float JuliaX;
        private float JuliaY;

        private bool[] Flags = new bool[4];
        private bool MandFlag;
        private bool[] QuatFlags = new bool[4];
        private bool[] JuliaFlags = new bool[2];

        public MainWindow()
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            InitializeComponent();
            ComboType.SelectedIndex = (int)(FractalType.Mand2D);
            BoxWinSize.Text = "500";
            BoxFractalSize.Text = "500";
            BoxIters.Text = "10";
            BoxMaxFractalSize.Text = "500";
            BoxCX.Text = "-0.512511498387847";
            BoxCY.Text = "0.521295573094847";
            BoxP.Text = "8.0";
            BoxQR.Text = "-0.65";
            BoxQA.Text = "-0.5";
            BoxQB.Text = "0.0";
            BoxQC.Text = "0.0";
        }

        private void ComboType_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            switch ((FractalType)ComboType.SelectedIndex)
            {
                case FractalType.Mand2D:
                    BlockD1.Text = "^ 2";
                    DockJulia.IsEnabled = false;
                    DockMand.IsEnabled = false;
                    DockQuat.IsEnabled = ComboComponent.IsEnabled = false;
                    break;
                case FractalType.Julia2D:
                    BlockD1.Text = "^ 2";
                    DockJulia.IsEnabled = true;
                    DockMand.IsEnabled = false;
                    DockQuat.IsEnabled = ComboComponent.IsEnabled = false;
                    break;
                case FractalType.Mand3D:
                    BlockD1.Text = "^ 3";
                    DockJulia.IsEnabled = false;
                    DockMand.IsEnabled = true;
                    DockQuat.IsEnabled = ComboComponent.IsEnabled = false;
                    break;
                case FractalType.Julia4D:
                    BlockD1.Text = "^ 3";
                    DockJulia.IsEnabled = false;
                    DockMand.IsEnabled = false;
                    DockQuat.IsEnabled = ComboComponent.IsEnabled = true;
                    break;
            }
            BlockD2.Text = BlockD1.Text;
        }

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

        private void BoxP_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                MandP = float.Parse(BoxP.Text);
                if (MandP < 2.0)
                    throw new Exception();
                BoxP.Background = new SolidColorBrush(Colors.White);
                MandFlag = true;
            }
            catch
            {
                BoxP.Background = new SolidColorBrush(Colors.Pink);
                MandFlag = false;
            }
        }

        private void BoxQR_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                QuatR = float.Parse(BoxQR.Text);
                BoxQR.Background = new SolidColorBrush(Colors.White);
                QuatFlags[0] = true;
            }
            catch
            {
                BoxQR.Background = new SolidColorBrush(Colors.Pink);
                QuatFlags[0] = false;
            }
        }

        private void BoxQA_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                QuatA = float.Parse(BoxQA.Text);
                BoxQA.Background = new SolidColorBrush(Colors.White);
                QuatFlags[1] = true;
            }
            catch
            {
                BoxQA.Background = new SolidColorBrush(Colors.Pink);
                QuatFlags[1] = false;
            }
        }

        private void BoxQB_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                QuatB = float.Parse(BoxQB.Text);
                BoxQB.Background = new SolidColorBrush(Colors.White);
                QuatFlags[2] = true;
            }
            catch
            {
                BoxQB.Background = new SolidColorBrush(Colors.Pink);
                QuatFlags[2] = false;
            }
        }

        private void BoxQC_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                QuatC = float.Parse(BoxQC.Text);
                BoxQC.Background = new SolidColorBrush(Colors.White);
                QuatFlags[3] = true;
            }
            catch
            {
                BoxQC.Background = new SolidColorBrush(Colors.Pink);
                QuatFlags[3] = false;
            }
        }

        private void BoxCX_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                JuliaX = float.Parse(BoxCX.Text);
                BoxCX.Background = new SolidColorBrush(Colors.White);
                JuliaFlags[0] = true;
            }
            catch
            {
                BoxCX.Background = new SolidColorBrush(Colors.Pink);
                JuliaFlags[0] = false;
            }
        }

        private void BoxCY_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                JuliaY = float.Parse(BoxCY.Text);
                BoxCY.Background = new SolidColorBrush(Colors.White);
                JuliaFlags[1] = true;
            }
            catch
            {
                BoxCY.Background = new SolidColorBrush(Colors.Pink);
                JuliaFlags[1] = false;
            }
        }

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

        private void ButtonRender_Click(object sender, RoutedEventArgs e)
        {
            if (!ValidationTest())
                return;
            //switch ((FractalType)ComboType.SelectedIndex)
            //{
            //    case FractalType.Mand2D:

            //        break;
            //}
        }
    }
}
