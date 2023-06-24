import mlflow


def calculate_sum(x,y):
    return x*y


if __name__ == '__main__':
    with mlflow.start_run():
        x,y=75,10
        z = calculate_sum(x,y)
        mlflow.log_param("X",x)
        mlflow.log_param("Y",y)
        mlflow.log_metric("Z",z)
