import numpy as np
import pandas as pd
import joblib
import base64
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression

import SessionState
import streamlit as st
import altair as alt

def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def pred_color(val):
    if val == 'Carrier':
        color_code = '#FFA500'
    elif val == 'ATL':
        color_code = '#0000FF'
    elif val == 'HAM':
        color_code = '#800080'
    elif val == 'Carrier_to_ATL':
        color_code = '#008000'
    elif val == 'HAM_Marianna':
        color_code = '#FF3366'

    r = int(color_code[1:3], 16)
    g = int(color_code[3:5], 16)
    b = int(color_code[5:7], 16)

    alpha = 0.5

    return f'background-color: rgba({r},{g},{b},{alpha})'

def logarithm(data):
    lst_transformed = []
    _, ncol = data.shape
    for i in range(0,ncol):
        tmp = data.iloc[:,i]
        tmp_transformed = np.log(tmp+1)
        lst_transformed.append(tmp_transformed)
    df_normalized = pd.DataFrame(lst_transformed).transpose()
    data = df_normalized

    return data

def my_round(val, digit=0):
    p = 10 ** digit
    return (val * p * 2 + 1) // 2 / p

def my_round2(val, digit=0):
    val_str = str(val)
    w = len(val_str) - digit
    zeros_ = '0'
    for _ in range(digit - 1):
        zeros_ += '0'
    return int(str(int('{}'.format(val)[:w]) + 1) + zeros_)

def calc_value_range(data):
    st.sidebar.header("Step2")


    max_values1 = int(my_round(max(data['PVL']), -1))
    min_values1 = math.floor(min(data['PVL']))
    max_values2 = int(my_round2(int(max(data['tax'])), 3))
    min_values2 = math.floor(min(data['tax']))
    max_values3 = int(my_round2(int(max(data['env'])), 3))
    min_values3 = math.floor(min(data['env']))
    max_values4 = int(my_round2(int(max(data['gag_p15'])), 3))
    min_values4 = math.floor(min(data['gag_p15']))
    max_values5 = int(my_round2(int(max(data['gag_p19'])), 3))
    min_values5 = math.floor(min(data['gag_p19']))
    max_values6 = int(my_round2(int(max(data['gag_p24'])), 3))
    min_values6 = math.floor(min(data['gag_p24']))

    values1 = st.sidebar.slider("PVL", 0, max_values1, (0, max_values1), 1)
    values2 = st.sidebar.slider("tax", 0, max_values2, (0, max_values2), 1000)
    values3 = st.sidebar.slider("env", 0, max_values3, (0, max_values3), 100)
    values4 = st.sidebar.slider("gag_p15", 0, max_values4, (0, max_values4), 1000)
    values5 = st.sidebar.slider("gag_p19", 0, max_values5, (0, max_values5), 1000)
    values6 = st.sidebar.slider("gag_p24", 0, max_values6, (0, max_values6), 1000)

    v1_1, v2_1 = values1
    v1_2, v2_2 = values2
    v1_3, v2_3 = values3
    v1_4, v2_4 = values4
    v1_5, v2_5 = values5
    v1_6, v2_6 = values6

    values1 = tuple(np.log((v1_1 + 1, v2_1 + 1)))
    values2 = tuple(np.log((v1_2 + 1, v2_2 + 1)))
    values3 = tuple(np.log((v1_3 + 1, v2_3 + 1)))
    values4 = tuple(np.log((v1_4 + 1, v2_4 + 1)))
    values5 = tuple(np.log((v1_5 + 1, v2_5 + 1)))
    values6 = tuple(np.log((v1_6 + 1, v2_6 + 1)))

    return values1, values2, values3, values4, values5, values6


def vasualizaiton_with_size(data, focal_name, domain, range):

    global figure_range_x, figure_range_y

    c = alt.Chart(data).mark_circle().encode(
    alt.X('axis01',
        scale=alt.Scale(domain=figure_range_x)
    ),
    alt.Y('axis02',
        scale=alt.Scale(domain=figure_range_y)
    ),
    color=alt.Color(focal_name, scale=alt.Scale(domain=domain, range=range), legend=alt.Legend(title="Color of the classes")),
    size="size",
    tooltip=['indivID', 'PVL', 'tax', 'env', 'gag_p15', 'gag_p19', 'gag_p24'],
    opacity = "opacity"
    ).properties(
        width=700,
        height=500
    ).interactive()

    return c


def vasualizaiton_without_size(data, focal_name, domain, range):

    global figure_range_x, figure_range_y

    c = alt.Chart(data).mark_circle(size=60).encode(
    alt.X('axis01',
        scale=alt.Scale(domain=figure_range_x)
    ),
    alt.Y('axis02',
        scale=alt.Scale(domain=figure_range_y)
    ),
    color=alt.Color(focal_name, scale=alt.Scale(domain=domain, range=range), legend=alt.Legend(title="Color of the classes")),
    tooltip=['indivID', 'PVL', 'tax', 'env', 'gag_p15', 'gag_p19', 'gag_p24']
    ).properties(
        width=700,
        height=500
    ).interactive()

    return c

st.set_page_config(layout="wide")


default_size = 15
emphasize_size = 90

default_opacity = 1.0
decrease_opacity = 0.3

st.header("upload csv file")

uploaded_file = st.file_uploader("file_upload", type='csv')

if uploaded_file == None:
    st.error('Data not uploaded')
else:

    sample_data = pd.read_csv(uploaded_file)
    sample_data['size'] = default_size
    sample_data['opacity'] = default_opacity

    variable_data = sample_data.loc[:,['PVL', 'tax', 'env', 'gag_p15', 'gag_p19', 'gag_p24']]
    non_variable_data = sample_data.loc[:,['indivID', 'type','size', 'opacity']]

    # logarize
    log_variable_data = logarithm(variable_data)
    sample_data = log_variable_data.join(non_variable_data, how = 'outer')
    figure_range_x = (-4.2, 4.2)
    figure_range_y = (-3, 6.5)


    data_processor = joblib.load('data_processor.pkl')

    pls_data = data_processor.transform(log_variable_data)
    pls_data = pd.DataFrame(pls_data)
    pls_data.columns = ['axis01','axis02']
    pls_data = pls_data.join(non_variable_data, how = 'outer').join(variable_data, how = 'outer')
    # pls_data = pls_data.join(variable_data, how = 'outer')

    print(pls_data.head())


    domain = ['Carrier','ATL','Carrier_to_ATL','HAM', 'HAM_Marianna']
    domain_color = {'Carrier':'#FFA500','ATL':'#0000FF','Carrier_to_ATL':'#008000','HAM':'#800080', 'HAM_Marianna':'#FF3366'}

    # color
    # orange : #FFA500
    # blue   : #0000FF
    # green  : #008000
    # purple : #800080
    # pink   : #FF3366


    all_range_ = ['#FFA500','#0000FF','#008000','#800080', '#FF3366']

    st.title("Step1 : Play with Data")

    options = st.multiselect('option1 : select individuals', list(pls_data['indivID']))

    if options == []:
        selected_targets = st.multiselect('option2 : select classes', domain, default=domain)

        re_pls_data = pls_data[pls_data['type'].isin(selected_targets)]

        range_ = []
        for name in selected_targets:
            range_.append(domain_color[name])

        fig1 = vasualizaiton_without_size(re_pls_data, 'type', selected_targets, range_)

        brush = alt.selection_interval()

        palette = alt.Scale(domain=['Carrier','ATL','Carrier_to_ATL','HAM', 'HAM_Marianna'],
                  range=['#FFA500','#0000FF','#008000','#800080', '#FF3366'])

        points = alt.Chart(re_pls_data).mark_circle(size=60).encode(
            alt.X('axis01',
            scale=alt.Scale(domain=figure_range_x)
            ),
            alt.Y('axis02',
            scale=alt.Scale(domain=figure_range_y)
            ),
            color=alt.condition(brush, 'type', alt.value('lightgray'), scale=palette),
            tooltip = ['indivID', 'PVL', 'tax', 'env', 'gag_p15', 'gag_p19', 'gag_p24']
        ).properties(
        width=500,
        height=300
        ).add_selection(
            brush
        )

        bars1 = alt.Chart(re_pls_data, title="PVL").mark_bar().encode(
            y='type',
            color='type',
            x='average(PVL)'
        ).properties(
                width=350,
                height=50
        ).transform_filter(
            brush
        )

        bars2 = alt.Chart(re_pls_data, title="tax").mark_bar().encode(
            y='type',
            color='type',
            x='average(tax)'
        ).properties(
                width=350,
                height=50
        ).transform_filter(
            brush
        )

        bars3 = alt.Chart(re_pls_data, title="env").mark_bar().encode(
            y='type',
            color='type',
            x='average(env)'
        ).properties(
                width=350,
                height=50
        ).transform_filter(
            brush
        )

        bars4 = alt.Chart(re_pls_data, title="gag_p15").mark_bar().encode(
            y='type',
            color='type',
            x='average(gag_p15)'
        ).properties(
                width=350,
                height=50
        ).transform_filter(
            brush
        )

        bars5 = alt.Chart(re_pls_data, title="gag_p19").mark_bar().encode(
            y='type',
            color='type',
            x='average(gag_p19)'
        ).properties(
                width=350,
                height=50
        ).transform_filter(
            brush
        )

        bars6 = alt.Chart(re_pls_data, title="gag_p24").mark_bar().encode(
            y='type',
            color='type',
            x='average(gag_p24)'
        ).properties(
                width=350,
                height=50
        ).transform_filter(
            brush
        )

        c1 = bars1 | bars2
        c2 = bars3 | bars4
        c3 = bars5 | bars6

        st.altair_chart(points & c1 & c2 & c3)

        # st.altair_chart(fig1)

    else:

        for name in options:
            pls_data.at[pls_data[pls_data['indivID'] == name].index[0], 'size'] = emphasize_size

        for name in list(pls_data['indivID']):
            if name not in options:
                pls_data.at[pls_data[pls_data['indivID'] == name].index[0], 'opacity'] = decrease_opacity

        fig2 = vasualizaiton_with_size(pls_data, 'type', domain, all_range_)
        st.altair_chart(fig2)

    st.title("Step2 : Data Preprocessing")
    st.header("delete data from Carrier, ATL, HAM")

    train_df = sample_data[sample_data['type'].isin(['Carrier','ATL','HAM'])]

    selected_targets = st.multiselect('select classes', ['Carrier','ATL','HAM'], default='Carrier')

    focal_train_df = train_df[train_df['type'].isin(selected_targets)]
    focal_pls_data = pls_data[pls_data['type'].isin(selected_targets)]

    re_pls_data = pls_data[pls_data['type'].isin(['Carrier','ATL','HAM'])]

    values1, values2, values3, values4, values5, values6 = calc_value_range(focal_pls_data)

    selected_train_df = focal_train_df[
    (focal_train_df['PVL'] >= values1[0]) & (focal_train_df['PVL'] <= values1[1]) & (focal_train_df['tax'] >= values2[0]) & (focal_train_df['tax'] <= values2[1]) &
    (focal_train_df['env'] >= values3[0]) & (focal_train_df['env'] <= values3[1]) & (focal_train_df['gag_p15'] >= values4[0]) & (focal_train_df['gag_p15'] <= values4[1]) &
    (focal_train_df['gag_p19'] >= values5[0]) & (focal_train_df['gag_p19'] <= values5[1]) & (focal_train_df['gag_p24'] >= values6[0]) & (focal_train_df['gag_p24'] <= values6[1])]

    variable_data = selected_train_df.loc[:,['PVL', 'tax', 'env', 'gag_p15', 'gag_p19', 'gag_p24']]
    non_variable_data = selected_train_df.loc[:,['indivID', 'type','size', 'opacity']]

    focal_ind = list(focal_train_df['indivID'])
    selected_ind = list(selected_train_df['indivID'])
    all_ind = list(re_pls_data['indivID'])

    selected_type = []
    for name in all_ind:
        if name in focal_ind:
            if name in selected_ind:
                selected_type.append(re_pls_data.at[re_pls_data[re_pls_data['indivID'] == name].index[0], 'type'])
            else:
                selected_type.append('not selected')
        else:
            selected_type.append(re_pls_data.at[re_pls_data[re_pls_data['indivID'] == name].index[0], 'type'])

    selected_domain = list(set(selected_type))
    selected_domain_color_ = []
    for ind_type in selected_domain:
        if ind_type in domain:
            selected_domain_color_.append(domain_color[ind_type])
        else:
            selected_domain_color_.append('#c9c9c9')

    re_pls_data['selected_type'] = selected_type

    for name in selected_ind:
        re_pls_data.at[re_pls_data[re_pls_data['indivID'] == name].index[0], 'size'] = emphasize_size

    for name in list(re_pls_data['indivID']):
        if name not in selected_ind:
            re_pls_data.at[re_pls_data[re_pls_data['indivID'] == name].index[0], 'opacity'] = decrease_opacity

    st.header("Used data : {}/{}".format(len(selected_ind), len(focal_ind)))

    fig3 = vasualizaiton_without_size(re_pls_data, 'selected_type', selected_domain, selected_domain_color_)
    st.altair_chart(fig3)


    model = RandomForestClassifier(max_depth=5,n_estimators=1000,max_features=3)

    train_df['type'] = selected_type
    train_df = train_df[train_df['type'] != 'not selected']

    train_y = []

    for i in range(len(train_df)):
        if train_df.iloc[i]['type'] == 'Carrier':
            train_y.append(0)
        elif train_df.iloc[i]['type'] == 'ATL':
            train_y.append(1)
        elif train_df.iloc[i]['type'] == 'HAM':
            train_y.append(2)

    st.title("Step3: Machine Learning")

    st.header("what data to use as test data?")

    selected_targets = st.multiselect('select classes', ['Carrier_to_ATL','HAM_Marianna'], default=['Carrier_to_ATL','HAM_Marianna'])

    test_df = sample_data[sample_data['type'].isin(selected_targets)]

    test_variable_data = test_df.loc[:,['PVL', 'tax', 'env', 'gag_p15', 'gag_p19', 'gag_p24']]
    test_non_variable_data = test_df.loc[:,['indivID', 'type','size', 'opacity']]

    answer1 = st.button('Training and Testing')

    session_state = SessionState.get(result = None)

    if answer1 == True:
        train_variable_data = train_df.loc[:,['PVL', 'tax', 'env', 'gag_p15', 'gag_p19', 'gag_p24']]
        clf = model.fit(train_variable_data.values, np.array(train_y))
        pred_num = clf.predict(test_variable_data.values)
        st.success("properly done")
        pred_name = []
        for i in pred_num:
            if i == 0:
                pred_name.append('Carrier')
            elif i == 1:
                pred_name.append('ATL')
            elif i == 2:
                pred_name.append('HAM')
        test_df['true_type'] = test_df['type']
        test_df['pred_type'] = pred_name

        session_state.result = test_df

    else:
         pass

    st.header("result")


    try:
        if session_state.result == None:
            st.error('not tested yet')

    except:

        options = st.multiselect('option1 : select individuals', list(test_df['indivID']))

        options_class = st.multiselect('option2 : select classes', ['Carrier_to_ATL', 'HAM_Marianna'])

        if options != []:

            pls_data['size'] = default_size
            pls_data['opacity'] = default_opacity

            selected_test_df = pd.DataFrame(columns = ['indivID', 'true', 'pred'])

            for name in options:
                pls_data.at[pls_data[pls_data['indivID'] == name].index[0], 'size'] = emphasize_size
                selected_test_df = selected_test_df.append({'indivID': name, 'true': session_state.result.at[session_state.result[session_state.result['indivID'] == name].index[0], 'true_type'], 'pred': session_state.result.at[session_state.result[session_state.result['indivID'] == name].index[0], 'pred_type']}, ignore_index=True)

            for name in list(pls_data['indivID']):
                if name not in options:
                    pls_data.at[pls_data[pls_data['indivID'] == name].index[0], 'opacity'] = decrease_opacity

            st.header("selected test data")

            col1, col2 = st.columns([1, 1.5])

            # fig4 = vasualizaiton_with_size(pls_data, 'type', domain, all_range_)

            temporal_name = []
            for name in list(pls_data['indivID']):
                if name in options:
                    temporal_name.append(session_state.result.at[session_state.result[session_state.result['indivID'] == name].index[0], 'pred_type'])
                else:
                    temporal_name.append('None')

            pls_data['pred'] = temporal_name

            fig4 = alt.Chart(pls_data).mark_circle().encode(
            alt.X('axis01',
                scale=alt.Scale(domain=figure_range_x)
            ),
            alt.Y('axis02',
                scale=alt.Scale(domain=figure_range_y)
            ),
            color=alt.Color('type', scale=alt.Scale(domain=domain, range=all_range_), legend=alt.Legend(title="Color of the classes")),
            size="size",
            tooltip=['indivID', 'PVL', 'tax', 'env', 'gag_p15', 'gag_p19', 'gag_p24', 'pred'],
            opacity = "opacity"
            ).properties(
                width=700,
                height=500
            ).interactive()


            col1.dataframe(selected_test_df.style.applymap(pred_color, subset=['pred']),width=370,height=len(selected_test_df)*50)
            col2.altair_chart(fig4)

        elif (options == []) and (options_class != []):

            pls_data['size'] = default_size
            pls_data['opacity'] = default_opacity

            selected_test_df = pd.DataFrame(columns = ['indivID', 'true', 'pred'])

            for ind_class in options_class:
                ind_class_pls_data = pls_data[pls_data['type'] == ind_class]
                ind_class_pls_data['true'] = session_state.result[session_state.result['type'] == ind_class]['true_type']
                ind_class_pls_data['pred'] = session_state.result[session_state.result['type'] == ind_class]['pred_type']
                ind_class_pls_data = ind_class_pls_data.reindex(columns=['indivID', 'true', 'pred'])
                selected_test_df = selected_test_df.append(ind_class_pls_data, ignore_index=True)
                for name in list(ind_class_pls_data['indivID']):
                    pls_data.at[pls_data[pls_data['indivID'] == name].index[0], 'size'] = emphasize_size

            for name in list(pls_data['indivID']):
                if pls_data.at[pls_data[pls_data['indivID'] == name].index[0], 'type'] not in options_class:
                    pls_data.at[pls_data[pls_data['indivID'] == name].index[0], 'opacity'] = decrease_opacity

            pred_class_count_df = pd.DataFrame(columns = ['true', 'pred', 'count'])
            selected_classes = list(set(selected_test_df['true']))

            for ind_class in selected_classes:
                pred_df = selected_test_df[selected_test_df['true'] == ind_class]
                pred_classes = list(set(pred_df['pred']))
                for name in pred_classes:
                    pred_class_count_df = pred_class_count_df.append({'true': ind_class, 'pred': name, 'count': len(pred_df[pred_df['pred'] == name])}, ignore_index=True)

            st.header("statistics")

            st.dataframe(pred_class_count_df,width=350,height=200)

            st.header("selected test data")

            file_name = st.text_input('enter file_name and download as CSV', '.csv')

            tmp_download_link = download_link(selected_test_df, file_name, 'Download as CSV')
            if file_name == '.csv':
                pass
            else:
                st.markdown(tmp_download_link, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1.5])


            temporal_name = []
            for name in list(pls_data['indivID']):
                if name in list(session_state.result['indivID']):
                    temporal_name.append(session_state.result.at[session_state.result[session_state.result['indivID'] == name].index[0], 'pred_type'])
                else:
                    temporal_name.append('None')

            pls_data['pred'] = temporal_name



            fig4 = alt.Chart(pls_data).mark_circle().encode(
            alt.X('axis01',
                scale=alt.Scale(domain=figure_range_x)
            ),
            alt.Y('axis02',
                scale=alt.Scale(domain=figure_range_y)
            ),
            color=alt.Color('type', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(title="Color of the classes")),
            size="size",
            tooltip=['indivID', 'PVL', 'tax', 'env', 'gag_p15', 'gag_p19', 'gag_p24', 'pred'],
            opacity = "opacity"
            ).properties(
                width=700,
                height=500
            ).interactive()


            # fig4 = vasualizaiton_with_size(pls_data, 'type', domain, all_range_)
            if len(selected_test_df) <= 24:
                col1.dataframe(selected_test_df.style.applymap(pred_color, subset=['pred']),width=370,height=len(selected_test_df)*50)
            else:
                col1.dataframe(selected_test_df.style.applymap(pred_color, subset=['pred']),width=370,height=12*50)
            col2.altair_chart(fig4)

        else:
            st.error('not selected yet')

