from pydantic_settings import BaseSettings

class AzureOpenAiSettings(BaseSettings):
  #各クラス変数の型を明示しないとFastAPI起動時にエラーになる。
  key: str = 'AZURE_OPENAI_API_KEY'
  version: str = '2024-02-01'
  endpoint: str = 'AZURE_OPENAI_ENDPOINT'
  model_name: str = 'REPLACE_WITH_YOUR_DEPLOYMENT_NAME'
  max_tokens: int = 10

#インポート先で毎回AzureOpenAiSettingsのインスタンスを生成する必要がないなら
#こちらでインスタンス生成した方が無駄な生成処理が減るはずである。
#しかしそういう目的ならlru_cacheを利用する方がいいのかもしれない。
#az_settings = AzureOpenAiSettings()
