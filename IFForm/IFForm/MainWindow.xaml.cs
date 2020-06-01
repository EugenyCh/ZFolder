using System;
using System.Collections.Generic;
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
        private bool[] Flags = new bool[4];

        public MainWindow()
        {
            InitializeComponent();
            ComboType.SelectedIndex = (int)(FractalType.Mand2D);
            BoxWinSize.Text = "500";
            BoxFractalSize.Text = "500";
            BoxIters.Text = "10";
            BoxMaxFractalSize.Text = "500";
        }

        private void ComboType_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            switch ((FractalType)ComboType.SelectedIndex)
            {
                case FractalType.Mand2D:
                case FractalType.Julia2D:
                    BlockD1.Text = "^ 2";
                    break;
                case FractalType.Mand3D:
                case FractalType.Julia4D:
                    BlockD1.Text = "^ 3";
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
    }
}
