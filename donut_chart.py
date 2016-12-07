from bokeh.charts import Donut, show, output_file
from bokeh.charts.utils import df_from_json
from bokeh.sampledata.olympics2014 import data
from bokeh.models import HoverTool
import pandas as pd

# utilize utility to make it easy to get json/dict data converted to a dataframe
#df = df_from_json(data)
df = pd.read_csv("chidata2016cleaned.tsv", sep='\t', encoding='utf-8')
#df.groupby(['primary_type', 'description']).count()
df = pd.DataFrame({'count': df.groupby(
    ["primary_type", "description"]).size()}).reset_index()
#
# filter by countries with at least one medal and sort by total medals
df['count2'] = pd.to_numeric(df['count'])
df = df[df['count'] > 5000]
df = df.sort("count", ascending=False)

#TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

# original example
d = Donut(df, label=['primary_type', 'description'], values='count2',
          text_font_size='8pt', hover_text='count')

#hover = d.select_one(HoverTool)
#hover.point_policy = "follow_mouse"
# hover.tooltips = [
#    ("Name", "df[count2]"),
#    ("Unemployment rate)", "@df[count2]"),
#    ("Count", "$count2"),
#]

output_file("donut.html", title="donut.py example")

show(d)
