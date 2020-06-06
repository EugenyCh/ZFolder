using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IFForm
{
    #region Template Classes
    public abstract class AIFTemplate
    {
        public string Name;
    }

    public class TemplateTMand2D : AIFTemplate
    {
        public float Power;
    }

    public class TemplateJulia2D : AIFTemplate
    {
        public float CX, CY;
    }

    public class TemplateTJulia2D : AIFTemplate
    {
        public float Power;
        public float CX, CY;
    }

    public class TemplateMand3D : AIFTemplate
    {
        public float Power;
    }

    public class TemplateTJulia3D : AIFTemplate
    {
        public float Power;
        public float CX, CY, CZ;
    }

    public class TemplateJulia4D : AIFTemplate
    {
        public int Hidden;
        public float CR, CX, CY, CZ;
    }

    public class TemplateTJulia4D : AIFTemplate
    {
        public int Power;
        public int Hidden;
        public float CR, CX, CY, CZ;
    }

    public class TemplateTMand4D : AIFTemplate
    {
        public int Power;
        public float CZ;
    }
    #endregion

    public class IFTemplates
    {
        public List<TemplateTMand2D> TMand2Ds = new List<TemplateTMand2D>();
        public List<TemplateJulia2D> Julia2Ds = new List<TemplateJulia2D>();
        public List<TemplateTJulia2D> TJulia2Ds = new List<TemplateTJulia2D>();
        public List<TemplateMand3D> Mand3Ds = new List<TemplateMand3D>();
        public List<TemplateTJulia3D> TJulia3Ds = new List<TemplateTJulia3D>();
        public List<TemplateJulia4D> Julia4Ds = new List<TemplateJulia4D>();
        public List<TemplateTJulia4D> TJulia4Ds = new List<TemplateTJulia4D>();
        public List<TemplateTMand4D> TMand4Ds = new List<TemplateTMand4D>();

        public void Add(TemplateTMand2D template) { TMand2Ds.Add(template); }
        public void Add(TemplateJulia2D template) { Julia2Ds.Add(template); }
        public void Add(TemplateTJulia2D template) { TJulia2Ds.Add(template); }
        public void Add(TemplateMand3D template) { Mand3Ds.Add(template); }
        public void Add(TemplateTJulia3D template) { TJulia3Ds.Add(template); }
        public void Add(TemplateJulia4D template) { Julia4Ds.Add(template); }
        public void Add(TemplateTJulia4D template) { TJulia4Ds.Add(template); }
        public void Add(TemplateTMand4D template) { TMand4Ds.Add(template); }
    }
}
