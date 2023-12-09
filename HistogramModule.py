from mesa.visualization.ModularVisualization import VisualizationElement, CHART_JS_FILE
from utils import histogram

class HistogramModule(VisualizationElement):
    package_includes = [CHART_JS_FILE]
    local_includes = ["HistogramModule.js"]
 
    def __init__(self, bins, label, canvas_height=200, canvas_width=500):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = bins
        self.data = None
        self.label = label
        new_element = "new HistogramModule({}, {}, {})"
        new_element = new_element.format([round(-1 + i*2/bins, 2) for i in range(bins)],
                                         canvas_width, 
                                         canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        self.data = model.datacollector.model_vars[self.label][-1] if len(model.datacollector.model_vars[self.label]) > 0 else []
        hist = histogram(self.data, bins=self.bins)
        return [x[0] for x in hist]
