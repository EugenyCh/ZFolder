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

        public MainWindow()
        {
            InitializeComponent();
            ComboType.SelectedIndex = (int)(FractalType.Mand2D);
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
    }
}
