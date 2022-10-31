def learning_curve(iterate_inf,xt,yt,xv,yv,model_base,param_name):
    from sklearn.metrics import mean_squared_error as mse
    import matplotlib.pyplot as plt
    train=[]
    test=[]
    if param_name=='max_depth' or param_name=='depth':
        for i in iterate_inf:
            try:
                model=model_base(max_depth=i,random_state=0)
            except NameError:
                model=model_base(depth=i,random_seed=0)
            model.fit(xt,yt)
            ypre_train=model.predict(xt)
            train.append(mse(yt,ypre_train))
            ypre_test=model.predict(xv)
            test.append(mse(yv,ypre_test))
    elif param_name=='learning_rate':
        for i in iterate_inf:
            model=model_base(learning_rate=i,random_state=0)
            model.fit(xt,yt)
            ypre_train=model.predict(xt)
            train.append(mse(yt,ypre_train))
            ypre_test=model.predict(xv)
            test.append(mse(yv,ypre_test))
    elif param_name=='iterations' or param_name=='max_iter':
        for i in iterate_inf:
            try:
                model=model_base(max_iter=i,random_state=0)
            except NameError:
                model=model_base(iterations=i)
            model.fit(xt,yt)
            ypre_train=model.predict(xt)
            train.append(mse(yt,ypre_train))
            ypre_test=model.predict(xv)
            test.append(mse(yv,ypre_test))
    elif param_name=='leaf_size' or param_name=='n_neighbors':
        for i in iterate_inf:
            try:
                model=model_base(n_jobs=-1,n_neighbors=i)
            except NameError:
                model=model_base(leaf_size=i,n_jobs=-1)
            model.fit(xt,yt)
            ypre_train=model.predict(xt)
            train.append(mse(yt,ypre_train))
            ypre_test=model.predict(xv)
            test.append(mse(yv,ypre_test))
    plt.figure(figsize=(10,10))
    plt.plot(iterate_inf,train,label='train',c='blue')
    plt.plot(iterate_inf,test,label='test',c='red')
    plt.xticks(iterate_inf)
    plt.show()
    return ''
def predict_power(model,df,target,r):
    from itertools import combinations
    from numpy import mean
    from sklearn.model_selection import cross_val_score
    if r==1:
        score= [(mean(cross_val_score(model,df[[x]],target,n_jobs=-1))).round(2) for x in df.columns]
        dict_scores=dict(zip(df.columns,score))
        return dict_scores
    elif r>1:
        for i in combinations(df.columns,r):
            res=(mean(cross_val_score(model,df[list(i)],target,n_jobs=-1))).round(2)
            print(f'{i}--->{res}')
        return 'end'
def outlier_finder(df):
    index_list=[]
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    iqr=q3-q1
    higher_bound=q3+1.5*iqr
    lower_bound=q1-1.5*iqr
    index_list.extend((df[(df<lower_bound) | (df>higher_bound)]).index)
    return index_list






















