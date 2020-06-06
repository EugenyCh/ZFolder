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
using System.Xml;

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
        private float Julia4DQR;
        private float Julia4DQA;
        private float Julia4DQB;
        private float Julia4DQC;
        private int TJulia4DP;
        private float TJulia4DQR;
        private float TJulia4DQA;
        private float TJulia4DQB;
        private float TJulia4DQC;
        private int TMand4DP;
        private float TMand4DQC;

        private bool[] FlagsMain = new bool[4] { true, true, true, true };
        private bool FlagTMand2DP = true;
        private bool FlagJulia2DCX = true;
        private bool FlagJulia2DCY = true;
        private bool FlagTJulia2DP = true;
        private bool FlagTJulia2DCX = true;
        private bool FlagTJulia2DCY = true;
        private bool FlagMand3DP = true;
        private bool FlagTJulia3DP = true;
        private bool FlagTJulia3DCX = true;
        private bool FlagTJulia3DCY = true;
        private bool FlagTJulia3DCZ = true;
        private bool FlagJulia4DQR = true;
        private bool FlagJulia4DQA = true;
        private bool FlagJulia4DQB = true;
        private bool FlagJulia4DQC = true;
        private bool FlagTJulia4DP = true;
        private bool FlagTJulia4DQR = true;
        private bool FlagTJulia4DQA = true;
        private bool FlagTJulia4DQB = true;
        private bool FlagTJulia4DQC = true;
        private bool FlagTMand4DP = true;
        private bool FlagTMand4DQC = true;

        private IFTemplates innerTemplates = new IFTemplates();
        private ConsoleColor DefaultFore;
        public MainWindow()
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            DefaultFore = Console.ForegroundColor;
            InitializeComponent();
            // XML Innner Reading
            try
            {
                XmlDocument xDoc = new XmlDocument();
                xDoc.Load("inner.xml");
                XmlElement xRoot = xDoc.DocumentElement;
                foreach (XmlNode xNode in xRoot)
                {
                    if (xNode is XmlElement)
                    {
                        XmlElement xfractal = (XmlElement)xNode;
                        string type = xfractal.Attributes.GetNamedItem("type").Value;
                        if (type == "TMand2D")
                        {
                            TemplateTMand2D template = new TemplateTMand2D
                            {
                                Name = xfractal.Attributes.GetNamedItem("name").Value,
                                Power = float.Parse(xfractal.GetElementsByTagName("power")[0].InnerText)
                            };
                            innerTemplates.Add(template);
                            Console.WriteLine($"Added [{type}] '{template.Name}'");
                            Console.WriteLine($"| Power = {template.Power}");
                        }
                        if (type == "Julia2D")
                        {
                            TemplateJulia2D template = new TemplateJulia2D
                            {
                                Name = xfractal.Attributes.GetNamedItem("name").Value,
                                CX = float.Parse(xfractal.GetElementsByTagName("cx")[0].InnerText),
                                CY = float.Parse(xfractal.GetElementsByTagName("cy")[0].InnerText)
                            };
                            innerTemplates.Add(template);
                            Console.WriteLine($"Added [{type}] '{template.Name}'");
                            Console.WriteLine($"| CX = {template.CX}");
                            Console.WriteLine($"| CY = {template.CY}");
                        }
                        if (type == "TJulia2D")
                        {
                            TemplateTJulia2D template = new TemplateTJulia2D
                            {
                                Name = xfractal.Attributes.GetNamedItem("name").Value,
                                Power = float.Parse(xfractal.GetElementsByTagName("power")[0].InnerText),
                                CX = float.Parse(xfractal.GetElementsByTagName("cx")[0].InnerText),
                                CY = float.Parse(xfractal.GetElementsByTagName("cy")[0].InnerText)
                            };
                            innerTemplates.Add(template);
                            Console.WriteLine($"Added [{type}] '{template.Name}'");
                            Console.WriteLine($"| Power = {template.Power}");
                            Console.WriteLine($"| CX = {template.CX}");
                            Console.WriteLine($"| CY = {template.CY}");
                        }
                        if (type == "Mand3D")
                        {
                            TemplateMand3D template = new TemplateMand3D
                            {
                                Name = xfractal.Attributes.GetNamedItem("name").Value,
                                Power = float.Parse(xfractal.GetElementsByTagName("power")[0].InnerText)
                            };
                            innerTemplates.Add(template);
                            Console.WriteLine($"Added [{type}] '{template.Name}'");
                            Console.WriteLine($"| Power = {template.Power}");
                        }
                        if (type == "TJulia3D")
                        {
                            TemplateTJulia3D template = new TemplateTJulia3D
                            {
                                Name = xfractal.Attributes.GetNamedItem("name").Value,
                                Power = float.Parse(xfractal.GetElementsByTagName("power")[0].InnerText),
                                CX = float.Parse(xfractal.GetElementsByTagName("cx")[0].InnerText),
                                CY = float.Parse(xfractal.GetElementsByTagName("cy")[0].InnerText),
                                CZ = float.Parse(xfractal.GetElementsByTagName("cz")[0].InnerText)
                            };
                            innerTemplates.Add(template);
                            Console.WriteLine($"Added [{type}] '{template.Name}'");
                            Console.WriteLine($"| Power = {template.Power}");
                            Console.WriteLine($"| CX = {template.CX}");
                            Console.WriteLine($"| CY = {template.CY}");
                            Console.WriteLine($"| CZ = {template.CZ}");
                        }
                        if (type == "Julia4D")
                        {
                            char ch = char.Parse(xfractal.GetElementsByTagName("h")[0].InnerText);
                            int ih = 0;
                            switch (ch)
                            {
                                case 'R': ih = 0; break;
                                case 'X': ih = 1; break;
                                case 'Y': ih = 2; break;
                                case 'Z': ih = 3; break;
                            }
                            TemplateJulia4D template = new TemplateJulia4D
                            {
                                Name = xfractal.Attributes.GetNamedItem("name").Value,
                                Hidden = ih,
                                CR = float.Parse(xfractal.GetElementsByTagName("cr")[0].InnerText),
                                CX = float.Parse(xfractal.GetElementsByTagName("cx")[0].InnerText),
                                CY = float.Parse(xfractal.GetElementsByTagName("cy")[0].InnerText),
                                CZ = float.Parse(xfractal.GetElementsByTagName("cz")[0].InnerText)
                            };
                            innerTemplates.Add(template);
                            Console.WriteLine($"Added [{type}] '{template.Name}'");
                            Console.WriteLine($"| CR = {template.CR}");
                            Console.WriteLine($"| CX = {template.CX}");
                            Console.WriteLine($"| CY = {template.CY}");
                            Console.WriteLine($"| CZ = {template.CZ}");
                            Console.WriteLine($"| Hidden = {ch} [{template.Hidden}]");
                        }
                        if (type == "TJulia4D")
                        {
                            char ch = char.Parse(xfractal.GetElementsByTagName("h")[0].InnerText);
                            int ih = 0;
                            switch (ch)
                            {
                                case 'R': ih = 0; break;
                                case 'X': ih = 1; break;
                                case 'Y': ih = 2; break;
                                case 'Z': ih = 3; break;
                            }
                            TemplateTJulia4D template = new TemplateTJulia4D
                            {
                                Name = xfractal.Attributes.GetNamedItem("name").Value,
                                Power = int.Parse(xfractal.GetElementsByTagName("power")[0].InnerText),
                                Hidden = ih,
                                CR = float.Parse(xfractal.GetElementsByTagName("cr")[0].InnerText),
                                CX = float.Parse(xfractal.GetElementsByTagName("cx")[0].InnerText),
                                CY = float.Parse(xfractal.GetElementsByTagName("cy")[0].InnerText),
                                CZ = float.Parse(xfractal.GetElementsByTagName("cz")[0].InnerText)
                            };
                            innerTemplates.Add(template);
                            Console.WriteLine($"Added [{type}] '{template.Name}'");
                            Console.WriteLine($"| Power = {template.Power}");
                            Console.WriteLine($"| CR = {template.CR}");
                            Console.WriteLine($"| CX = {template.CX}");
                            Console.WriteLine($"| CY = {template.CY}");
                            Console.WriteLine($"| CZ = {template.CZ}");
                            Console.WriteLine($"| Hidden = {ch} [{template.Hidden}]");
                        }
                        if (type == "TMand4D")
                        {
                            TemplateTMand4D template = new TemplateTMand4D
                            {
                                Name = xfractal.Attributes.GetNamedItem("name").Value,
                                Power = int.Parse(xfractal.GetElementsByTagName("power")[0].InnerText),
                                CZ = float.Parse(xfractal.GetElementsByTagName("cz")[0].InnerText)
                            };
                            innerTemplates.Add(template);
                            Console.WriteLine($"Added [{type}] '{template.Name}'");
                            Console.WriteLine($"| Power = {template.Power}");
                            Console.WriteLine($"| CZ = {template.CZ}");
                        }
                    }
                }

                #region TMand2Ds
                MenuItem itemTMand2Ds = new MenuItem
                {
                    Header = "Комплексный фрактал типа Мандельброта 2D"
                };
                ItemInner.Items.Add(itemTMand2Ds);
                foreach (var fractal in innerTemplates.TMand2Ds)
                {
                    MenuItem item = new MenuItem
                    {
                        Header = fractal.Name
                    };
                    MenuItem subItem = new MenuItem
                    {
                        Header = $"z -> z ^ {fractal.Power} + c\nc = C"
                    };
                    item.Items.Add(subItem);
                    subItem.Click += Item_Click;
                    itemTMand2Ds.Items.Add(item);
                }
                #endregion

                #region TMand2Ds
                MenuItem itemsJulia2Ds = new MenuItem
                {
                    Header = "Комплексный фрактал Жюлиа 2D"
                };
                ItemInner.Items.Add(itemsJulia2Ds);
                foreach (var fractal in innerTemplates.Julia2Ds)
                {
                    MenuItem item = new MenuItem
                    {
                        Header = fractal.Name
                    };
                    MenuItem subItem = new MenuItem
                    {
                        Header = $"z -> z ^ 2 + c\nc = {fractal.CX} + {fractal.CY}i"
                    };
                    item.Items.Add(subItem);
                    subItem.Click += Item_Click;
                    itemsJulia2Ds.Items.Add(item);
                }
                #endregion

                #region TJulia2Ds
                MenuItem itemsTJulia2Ds = new MenuItem
                {
                    Header = "Комплексный фрактал типа Жюлиа 2D"
                };
                ItemInner.Items.Add(itemsTJulia2Ds);
                foreach (var fractal in innerTemplates.TJulia2Ds)
                {
                    MenuItem item = new MenuItem
                    {
                        Header = fractal.Name
                    };
                    MenuItem subItem = new MenuItem
                    {
                        Header = $"z -> z ^ {fractal.Power} + c\nc = {fractal.CX} + {fractal.CY}i"
                    };
                    item.Items.Add(subItem);
                    subItem.Click += Item_Click;
                    itemsTJulia2Ds.Items.Add(item);
                }
                #endregion

                #region Mand3Ds
                MenuItem itemsMand3Ds = new MenuItem
                {
                    Header = "Гиперкомплексная оболочка Мандельброта 3D"
                };
                ItemInner.Items.Add(itemsMand3Ds);
                foreach (var fractal in innerTemplates.Mand3Ds)
                {
                    MenuItem item = new MenuItem
                    {
                        Header = fractal.Name
                    };
                    MenuItem subItem = new MenuItem
                    {
                        Header = $"v -> v ^ {fractal.Power} + c\nc = R^3"
                    };
                    item.Items.Add(subItem);
                    subItem.Click += Item_Click;
                    itemsMand3Ds.Items.Add(item);
                }
                #endregion

                #region TJulia3Ds
                MenuItem itemsTJulia3Ds = new MenuItem
                {
                    Header = "Гиперкомплексная оболочка типа Жюлиа 3D"
                };
                ItemInner.Items.Add(itemsTJulia3Ds);
                foreach (var fractal in innerTemplates.TJulia3Ds)
                {
                    MenuItem item = new MenuItem
                    {
                        Header = fractal.Name
                    };
                    MenuItem subItem = new MenuItem
                    {
                        Header = $"v -> v ^ {fractal.Power} + c\nc = ({fractal.CX}, {fractal.CY}, {fractal.CZ})"
                    };
                    item.Items.Add(subItem);
                    subItem.Click += Item_Click;
                    itemsTJulia3Ds.Items.Add(item);
                }
                #endregion

                #region Julia4Ds
                MenuItem itemsJulia4Ds = new MenuItem
                {
                    Header = "Кватернионный фрактал Жюлиа 4D"
                };
                ItemInner.Items.Add(itemsJulia4Ds);
                foreach (var fractal in innerTemplates.Julia4Ds)
                {
                    MenuItem item = new MenuItem
                    {
                        Header = fractal.Name
                    };
                    MenuItem subItem = new MenuItem
                    {
                        Header = $"q -> q ^ 2 + c\nc = {fractal.CR} + {fractal.CX}i + {fractal.CY}j + {fractal.CZ}k\n{fractal.Hidden}'th component is hidden"
                    };
                    item.Items.Add(subItem);
                    subItem.Click += Item_Click;
                    itemsJulia4Ds.Items.Add(item);
                }
                #endregion

                #region TJulia4Ds
                MenuItem itemsTJulia4Ds = new MenuItem
                {
                    Header = "Кватернионный фрактал типа Жюлиа 4D"
                };
                ItemInner.Items.Add(itemsTJulia4Ds);
                foreach (var fractal in innerTemplates.TJulia4Ds)
                {
                    MenuItem item = new MenuItem
                    {
                        Header = fractal.Name
                    };
                    MenuItem subItem = new MenuItem
                    {
                        Header = $"q -> q ^ {fractal.Power} + c\nc = {fractal.CR} + {fractal.CX}i + {fractal.CY}j + {fractal.CZ}k\n{fractal.Hidden}'th component is hidden"
                    };
                    item.Items.Add(subItem);
                    subItem.Click += Item_Click;
                    itemsTJulia4Ds.Items.Add(item);
                }
                #endregion

                #region TMand4Ds
                MenuItem itemsTMand4Ds = new MenuItem
                {
                    Header = "Кватернионный фрактал типа Мандельброта 4D"
                };
                ItemInner.Items.Add(itemsTMand4Ds);
                foreach (var fractal in innerTemplates.TMand4Ds)
                {
                    MenuItem item = new MenuItem
                    {
                        Header = fractal.Name
                    };
                    MenuItem subItem = new MenuItem
                    {
                        Header = $"q -> q ^ {fractal.Power} + c\nc = R + Ri + Rj + {fractal.CZ}"
                    };
                    item.Items.Add(subItem);
                    subItem.Click += Item_Click;
                    itemsTMand4Ds.Items.Add(item);
                }
                #endregion
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка при инициалищации встроенных шаблонов:\n{ex}",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            // Left Panel
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
            BoxTJulia2DCX.Text = "0.535";
            BoxTJulia2DCY.Text = "0.35";
            // Mand3D
            BoxMand3DP.Text = "8.0";
            // TJulia3D
            BoxTJulia3DP.Text = "4.0";
            BoxTJulia3DCX.Text = "0.45";
            BoxTJulia3DCY.Text = "0.50";
            BoxTJulia3DCZ.Text = "0.55";
            // Julia4D
            BoxJulia4DQR.Text = "-0.65";
            BoxJulia4DQA.Text = "-0.5";
            BoxJulia4DQB.Text = "0.0";
            BoxJulia4DQC.Text = "0.0";
            // TJulia4D
            BoxTJulia4DP.Text = "4";
            BoxTJulia4DQR.Text = "-1.0";
            BoxTJulia4DQA.Text = "0.2";
            BoxTJulia4DQB.Text = "0.0";
            BoxTJulia4DQC.Text = "0.0";
            // TMand4D
            BoxTMand4DP.Text = "2";
            BoxTMand4DQC.Text = "0.0";
            // Combo Gradient
            Combo2D.SelectedIndex = 0;
            Combo3D.SelectedIndex = 1;
            Combo4D.SelectedIndex = 1;
        }

        private void Item_Click(object sender, RoutedEventArgs e)
        {
            Console.WriteLine("Click");
            MenuItem parent = (MenuItem)((MenuItem)sender).Parent;
            MenuItem greatParent = (MenuItem)parent.Parent;
            int id = ItemInner.Items.IndexOf(greatParent) + 1;
            ComboType.SelectedIndex = id;
            int parentId = greatParent.Items.IndexOf(parent);
            switch ((FractalType)id)
            {
                case FractalType.TMand2D:
                    var template1 = innerTemplates.TMand2Ds[parentId];
                    BoxTMand2DP.Text = template1.Power.ToString();
                    break;
                case FractalType.Julia2D:
                    var template2 = innerTemplates.Julia2Ds[parentId];
                    BoxJulia2DCX.Text = template2.CX.ToString();
                    BoxJulia2DCY.Text = template2.CY.ToString();
                    break;
                case FractalType.TJulia2D:
                    var template3 = innerTemplates.TJulia2Ds[parentId];
                    BoxTJulia2DP.Text = template3.Power.ToString();
                    BoxTJulia2DCX.Text = template3.CX.ToString();
                    BoxTJulia2DCY.Text = template3.CY.ToString();
                    break;
                case FractalType.Mand3D:
                    var template4 = innerTemplates.Mand3Ds[parentId];
                    BoxMand3DP.Text = template4.Power.ToString();
                    break;
                case FractalType.TJulia3D:
                    var template5 = innerTemplates.TJulia3Ds[parentId];
                    BoxTJulia3DP.Text = template5.Power.ToString();
                    BoxTJulia3DCX.Text = template5.CX.ToString();
                    BoxTJulia3DCY.Text = template5.CY.ToString();
                    BoxTJulia3DCZ.Text = template5.CZ.ToString();
                    break;
                case FractalType.Julia4D:
                    var template6 = innerTemplates.Julia4Ds[parentId];
                    BoxJulia4DQR.Text = template6.CR.ToString();
                    BoxJulia4DQA.Text = template6.CX.ToString();
                    BoxJulia4DQB.Text = template6.CY.ToString();
                    BoxJulia4DQC.Text = template6.CZ.ToString();
                    ComboJulia4DComponent.SelectedIndex = template6.Hidden;
                    break;
                case FractalType.TJulia4D:
                    var template7 = innerTemplates.TJulia4Ds[parentId];
                    BoxTJulia4DP.Text = template7.Power.ToString();
                    BoxTJulia4DQR.Text = template7.CR.ToString();
                    BoxTJulia4DQA.Text = template7.CX.ToString();
                    BoxTJulia4DQB.Text = template7.CY.ToString();
                    BoxTJulia4DQC.Text = template7.CZ.ToString();
                    ComboTJulia4DComponent.SelectedIndex = template7.Hidden;
                    break;
                case FractalType.TMand4D:
                    var template8 = innerTemplates.TMand4Ds[parentId];
                    BoxTMand4DP.Text = template8.Power.ToString();
                    BoxTMand4DQC.Text = template8.CZ.ToString();
                    break;
            }
        }

        #region Fractal Type Selection
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
        #endregion

        #region Left Panel
        private void BoxWinSize_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                WindowSize = int.Parse(BoxWinSize.Text);
                if (WindowSize < 100)
                    throw new Exception();
                BoxWinSize.Background = new SolidColorBrush(Colors.White);
                FlagsMain[0] = true;
            }
            catch
            {
                BoxWinSize.Background = new SolidColorBrush(Colors.Pink);
                FlagsMain[0] = false;
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
                FlagsMain[1] = true;
            }
            catch
            {
                BoxFractalSize.Background = new SolidColorBrush(Colors.Pink);
                FlagsMain[1] = false;
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
                FlagsMain[2] = true;
            }
            catch
            {
                BoxIters.Background = new SolidColorBrush(Colors.Pink);
                FlagsMain[2] = false;
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
                FlagsMain[3] = true;
            }
            catch
            {
                BoxMaxFractalSize.Background = new SolidColorBrush(Colors.Pink);
                FlagsMain[3] = false;
            }
        }
        #endregion

        #region TMand2D
        private void BoxTMand2DP_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TMand2DP = float.Parse(BoxTMand2DP.Text);
                if (TMand2DP < 2.0)
                    throw new Exception();
                BoxTMand2DP.Background = new SolidColorBrush(Colors.White);
                FlagTMand2DP = true;
            }
            catch
            {
                BoxTMand2DP.Background = new SolidColorBrush(Colors.Pink);
                FlagTMand2DP = false;
            }
        }
        #endregion

        #region Julia2D
        private void BoxJulia2DCX_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Julia2DCX = float.Parse(BoxJulia2DCX.Text);
                BoxJulia2DCX.Background = new SolidColorBrush(Colors.White);
                FlagJulia2DCX = true;
            }
            catch
            {
                BoxJulia2DCX.Background = new SolidColorBrush(Colors.Pink);
                FlagJulia2DCX = false;
            }
        }

        private void BoxJulia2DCY_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Julia2DCY = float.Parse(BoxJulia2DCY.Text);
                BoxJulia2DCY.Background = new SolidColorBrush(Colors.White);
                FlagJulia2DCY = true;
            }
            catch
            {
                BoxJulia2DCY.Background = new SolidColorBrush(Colors.Pink);
                FlagJulia2DCY = false;
            }
        }
        #endregion

        #region TJulia2D
        private void BoxTJulia2DP_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia2DP = float.Parse(BoxTJulia2DP.Text);
                if (TJulia2DP < 2.0)
                    throw new Exception();
                BoxTJulia2DP.Background = new SolidColorBrush(Colors.White);
                FlagTJulia2DP = true;
            }
            catch
            {
                BoxTJulia2DP.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia2DP = false;
            }
        }
        private void BoxTJulia2DCX_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia2DCX = float.Parse(BoxTJulia2DCX.Text);
                BoxTJulia2DCX.Background = new SolidColorBrush(Colors.White);
                FlagTJulia2DCX = true;
            }
            catch
            {
                BoxTJulia2DCX.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia2DCX = false;
            }
        }

        private void BoxTJulia2DCY_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia2DCY = float.Parse(BoxTJulia2DCY.Text);
                BoxTJulia2DCY.Background = new SolidColorBrush(Colors.White);
                FlagTJulia2DCY = true;
            }
            catch
            {
                BoxTJulia2DCY.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia2DCY = false;
            }
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
                FlagMand3DP = true;
            }
            catch
            {
                BoxMand3DP.Background = new SolidColorBrush(Colors.Pink);
                FlagMand3DP = false;
            }
        }
        #endregion

        #region TJulia3D
        private void BoxTJulia3DCX_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia3DCX = float.Parse(BoxTJulia3DCX.Text);
                BoxTJulia3DCX.Background = new SolidColorBrush(Colors.White);
                FlagTJulia3DCX = true;
            }
            catch
            {
                BoxTJulia3DCX.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia3DCX = false;
            }
        }

        private void BoxTJulia3DCY_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia3DCY = float.Parse(BoxTJulia3DCY.Text);
                BoxTJulia3DCY.Background = new SolidColorBrush(Colors.White);
                FlagTJulia3DCY = true;
            }
            catch
            {
                BoxTJulia3DCY.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia3DCY = false;
            }
        }

        private void BoxTJulia3DCZ_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia3DCZ = float.Parse(BoxTJulia3DCZ.Text);
                BoxTJulia3DCZ.Background = new SolidColorBrush(Colors.White);
                FlagTJulia3DCZ = true;
            }
            catch
            {
                BoxTJulia3DCZ.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia3DCZ = false;
            }
        }

        private void BoxTJulia3DP_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia3DP = float.Parse(BoxTJulia3DP.Text);
                if (TJulia3DP < 2.0)
                    throw new Exception();
                BoxTJulia3DP.Background = new SolidColorBrush(Colors.White);
                FlagTJulia3DP = true;
            }
            catch
            {
                BoxTJulia3DP.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia3DP = false;
            }
        }
        #endregion

        #region Julia4D
        private void BoxJulia4DQR_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Julia4DQR = float.Parse(BoxJulia4DQR.Text);
                BoxJulia4DQR.Background = new SolidColorBrush(Colors.White);
                FlagJulia4DQR = true;
            }
            catch
            {
                BoxJulia4DQR.Background = new SolidColorBrush(Colors.Pink);
                FlagJulia4DQR = false;
            }
        }

        private void BoxJulia4DQA_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Julia4DQA = float.Parse(BoxJulia4DQA.Text);
                BoxJulia4DQA.Background = new SolidColorBrush(Colors.White);
                FlagJulia4DQA = true;
            }
            catch
            {
                BoxJulia4DQA.Background = new SolidColorBrush(Colors.Pink);
                FlagJulia4DQA = false;
            }
        }

        private void BoxJulia4DQB_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Julia4DQB = float.Parse(BoxJulia4DQB.Text);
                BoxJulia4DQB.Background = new SolidColorBrush(Colors.White);
                FlagJulia4DQB = true;
            }
            catch
            {
                BoxJulia4DQB.Background = new SolidColorBrush(Colors.Pink);
                FlagJulia4DQB = false;
            }
        }

        private void BoxJulia4DQC_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                Julia4DQC = float.Parse(BoxJulia4DQC.Text);
                BoxJulia4DQC.Background = new SolidColorBrush(Colors.White);
                FlagJulia4DQC = true;
            }
            catch
            {
                BoxJulia4DQC.Background = new SolidColorBrush(Colors.Pink);
                FlagJulia4DQC = false;
            }
        }
        #endregion

        #region TJulia4D
        private void BoxTJulia4DQR_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia4DQR = float.Parse(BoxTJulia4DQR.Text);
                BoxTJulia4DQR.Background = new SolidColorBrush(Colors.White);
                FlagTJulia4DQR = true;
            }
            catch
            {
                BoxTJulia4DQR.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia4DQR = false;
            }
        }

        private void BoxTJulia4DQA_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia4DQA = float.Parse(BoxTJulia4DQA.Text);
                BoxTJulia4DQA.Background = new SolidColorBrush(Colors.White);
                FlagTJulia4DQA = true;
            }
            catch
            {
                BoxTJulia4DQA.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia4DQA = false;
            }
        }

        private void BoxTJulia4DQB_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia4DQB = float.Parse(BoxTJulia4DQB.Text);
                BoxTJulia4DQB.Background = new SolidColorBrush(Colors.White);
                FlagTJulia4DQB = true;
            }
            catch
            {
                BoxTJulia4DQB.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia4DQB = false;
            }
        }

        private void BoxTJulia4DQC_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia4DQC = float.Parse(BoxTJulia4DQC.Text);
                BoxTJulia4DQC.Background = new SolidColorBrush(Colors.White);
                FlagTJulia4DQC = true;
            }
            catch
            {
                BoxTJulia4DQC.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia4DQC = false;
            }
        }

        private void BoxTJulia4DP_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TJulia4DP = int.Parse(BoxTJulia4DP.Text);
                if (TJulia4DP < 2)
                    throw new Exception();
                BoxTJulia4DP.Background = new SolidColorBrush(Colors.White);
                FlagTJulia4DP = true;
            }
            catch
            {
                BoxTJulia4DP.Background = new SolidColorBrush(Colors.Pink);
                FlagTJulia4DP = false;
            }
        }
        #endregion

        #region TMand4D
        private void BoxTMand4DQC_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TMand4DQC = float.Parse(BoxTMand4DQC.Text);
                BoxTMand4DQC.Background = new SolidColorBrush(Colors.White);
                FlagTMand4DQC = true;
            }
            catch
            {
                BoxTMand4DQC.Background = new SolidColorBrush(Colors.Pink);
                FlagTMand4DQC = false;
            }
        }

        private void BoxTMand4DP_TextChanged(object sender, TextChangedEventArgs e)
        {
            try
            {
                TMand4DP = int.Parse(BoxTMand4DP.Text);
                if (TMand4DP < 2.0)
                    throw new Exception();
                BoxTMand4DP.Background = new SolidColorBrush(Colors.White);
                FlagTMand4DP = true;
            }
            catch
            {
                BoxTMand4DP.Background = new SolidColorBrush(Colors.Pink);
                FlagTMand4DP = false;
            }
        }
        #endregion

        private bool ValidationTest()
        {
            #region Main Flags
            if (FlagsMain[0] == false)
            {
                MessageBox.Show("Ошибка при заполнении размера окна: должно быть целое число >= 100",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (FlagsMain[1] == false)
            {
                MessageBox.Show("Ошибка при заполнении размера фрактала: должно быть целое число >= 100",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (FlagsMain[2] == false)
            {
                MessageBox.Show("Ошибка при заполнении количества итераций: должно быть целое число >= 0",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (FlagsMain[3] == false)
            {
                MessageBox.Show("Ошибка при заполнении максимального размера фрактала: должно быть целое число >= 100",
                    "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            #endregion

            #region TMand2D Flags
            if ((FractalType)ComboType.SelectedIndex == FractalType.TMand2D)
            {
                if (FlagTMand2DP == false)
                {
                    MessageBox.Show("Ошибка при заполнении степени оболочки: должно быть число >= 2.0",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
            }
            #endregion

            #region Julia2D Flags
            if ((FractalType)ComboType.SelectedIndex == FractalType.Julia2D)
            {
                if (FlagJulia2DCX == false)
                {
                    MessageBox.Show("Ошибка при заполнении компоненты X в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagJulia2DCY == false)
                {
                    MessageBox.Show("Ошибка при заполнении компоненты Y в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
            }
            #endregion

            #region TJulia2D Flags
            if ((FractalType)ComboType.SelectedIndex == FractalType.TJulia2D)
            {
                if (FlagTJulia2DP == false)
                {
                    MessageBox.Show("Ошибка при заполнении степени: должно быть число >= 2.0",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTJulia2DCX == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты X в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTJulia2DCY == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты Y в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
            }
            #endregion

            #region Mand3D Flags
            if ((FractalType)ComboType.SelectedIndex == FractalType.Mand3D)
            {
                if (FlagMand3DP == false)
                {
                    MessageBox.Show("Ошибка при заполнении степени: должно быть число >= 2.0",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
            }
            #endregion

            #region TJulia3D Flags
            if ((FractalType)ComboType.SelectedIndex == FractalType.TJulia3D)
            {
                if (FlagTJulia3DP == false)
                {
                    MessageBox.Show("Ошибка при заполнении степени: должно быть число >= 2.0",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTJulia3DCX == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты X в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTJulia3DCY == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты Y в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTJulia3DCZ == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты Z в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
            }
            #endregion

            #region Julia4D Flags
            if ((FractalType)ComboType.SelectedIndex == FractalType.Julia4D)
            {
                if (FlagJulia4DQR == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты R в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagJulia4DQA == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты X в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagJulia4DQB == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты Y в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagJulia4DQC == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты Z в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
            }
            #endregion

            #region TJulia4D Flags
            if ((FractalType)ComboType.SelectedIndex == FractalType.TJulia4D)
            {
                if (FlagTJulia4DP == false)
                {
                    MessageBox.Show("Ошибка при заполнении степени: должно быть целое число >= 2",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTJulia4DQR == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты R в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTJulia4DQA == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты X в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTJulia4DQB == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты Y в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTJulia4DQC == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты Z в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
            }
            #endregion

            #region TMand4D Flags
            if ((FractalType)ComboType.SelectedIndex == FractalType.TMand4D)
            {
                if (FlagTMand4DP == false)
                {
                    MessageBox.Show("Ошибка при заполнении степени: должно быть целое число >= 2",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
                if (FlagTMand4DQC == false)
                {
                    MessageBox.Show("Ошибка при заполнении компненты Z в константе: должно быть число",
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }
            }
            #endregion

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
                    //Mand2D = 0,
                    //TMand2D = 1,
                    //Julia2D = 2,
                    //TJulia2D = 3,
                    //Mand3D = 4,
                    //TJulia3D = 5,
                    //Julia4D = 6,
                    //TJulia4D = 7,
                    //TMand4D = 8
                    switch ((FractalType)ComboType.SelectedIndex)
                    {
                        case FractalType.Mand2D:
                            WriteInt(writer, Combo2D.SelectedIndex);
                            WriteInt(writer, 0);
                            Console.WriteLine("Type: Mandelbrot 2D");
                            break;
                        case FractalType.TMand2D:
                            WriteInt(writer, Combo2D.SelectedIndex);
                            WriteInt(writer, 1);
                            WriteFloat(writer, TMand2DP);
                            Console.WriteLine("Type: T Mandelbrot 2D");
                            break;
                        case FractalType.Julia2D:
                            WriteInt(writer, Combo2D.SelectedIndex);
                            WriteInt(writer, 0);
                            WriteFloat(writer, Julia2DCX);
                            WriteFloat(writer, Julia2DCY);
                            Console.WriteLine("Type: Julia 2D");
                            break;
                        case FractalType.TJulia2D:
                            WriteInt(writer, Combo2D.SelectedIndex);
                            WriteInt(writer, 1);
                            WriteFloat(writer, TJulia2DCX);
                            WriteFloat(writer, TJulia2DCY);
                            WriteFloat(writer, TJulia2DP);
                            Console.WriteLine("Type: T Julia 2D");
                            break;
                        case FractalType.Mand3D:
                            WriteInt(writer, Combo3D.SelectedIndex);
                            WriteInt(writer, 0);
                            WriteFloat(writer, Mand3DP);
                            Console.WriteLine("Type: Mandelbulb 3D");
                            break;
                        case FractalType.TJulia3D:
                            WriteInt(writer, Combo3D.SelectedIndex);
                            WriteInt(writer, 1);
                            WriteFloat(writer, TJulia3DP);
                            WriteFloat(writer, TJulia3DCX);
                            WriteFloat(writer, TJulia3DCY);
                            WriteFloat(writer, TJulia3DCZ);
                            Console.WriteLine("Type: T Julia 3D");
                            break;
                        case FractalType.Julia4D:
                            WriteInt(writer, Combo4D.SelectedIndex);
                            WriteInt(writer, 0);
                            WriteFloat(writer, Julia4DQR);
                            WriteFloat(writer, Julia4DQA);
                            WriteFloat(writer, Julia4DQB);
                            WriteFloat(writer, Julia4DQC);
                            WriteInt(writer, ComboJulia4DComponent.SelectedIndex);
                            Console.WriteLine("Type: Julia 4D");
                            break;
                        case FractalType.TJulia4D:
                            WriteInt(writer, Combo4D.SelectedIndex);
                            WriteInt(writer, 1);
                            WriteFloat(writer, TJulia4DQR);
                            WriteFloat(writer, TJulia4DQA);
                            WriteFloat(writer, TJulia4DQB);
                            WriteFloat(writer, TJulia4DQC);
                            WriteInt(writer, ComboTJulia4DComponent.SelectedIndex);
                            WriteInt(writer, TJulia4DP);
                            Console.WriteLine("Type: T Julia 4D");
                            break;
                        case FractalType.TMand4D:
                            WriteInt(writer, Combo4D.SelectedIndex);
                            WriteInt(writer, 2);
                            WriteFloat(writer, TMand4DQC);
                            WriteInt(writer, TMand4DP);
                            Console.WriteLine("Type: T Mandelbrot 4D");
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
                    case FractalType.TMand2D:
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
                    case FractalType.TJulia2D:
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
                    case FractalType.TJulia3D:
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
                    case FractalType.TJulia4D:
                    case FractalType.TMand4D:
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
