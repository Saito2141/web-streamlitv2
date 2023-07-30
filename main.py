import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.title('Streamlit 超入門')

st.write('Display Image')

img = Image.open('layer_map.png')

st.image(img, caption='results', use_column_width=True)
# df = pd.DataFrame(
#     np.random.rand(100, 2)/[50,50] + [35.69, 139.70],
#     columns=['lat', 'lon']
# )

# 列を指定するときは，axis=0，行を指定するときはaxis=1
# st.table(df.style.highlight_max(axis=0))
# st.map(df)
# """
# # 章
# ## 節
# ### 項
#
# ```python
# import streamlit as st
# import numpy as np
# import pandas as pd
# ```
# """