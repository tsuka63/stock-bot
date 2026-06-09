# クラウド自動化セットアップ（GitHub Actions）

Macを起動していなくても、GitHubのサーバーが**毎日16:00(JST)に自動でランキングを取得**します。

---

## 前提

- GitHubアカウント（無料）。なければ https://github.com で作成。
- リポジトリは **public** にします（無料でGitHub Pages・Actionsを使うため）。
  - ⚠️ 公開されるのは「コードとランキング」だけ。
  - ✅ **保有銘柄・ポートフォリオ・パスワードは `.gitignore` で除外済み**なので**公開されません**。

---

## 手順（GitHub Desktop を使う・一番簡単）

### 1. GitHub Desktop をインストール
https://desktop.github.com からダウンロードしてサインイン。

### 2. このフォルダをリポジトリとして追加
- `File → Add Local Repository`
- `/Users/tsuka/Desktop/claude code/stock-bot` を選択

### 3. 公開する
- `Publish repository` をクリック
- 名前は任意（例: `stock-bot`）
- **「Keep this code private」のチェックは外す**（public）
- `Publish Repository`

### 4. GitHub Pages を有効化（レポートをWebで見る）
github.com の自分のリポジトリで:
- `Settings → Pages`
- Source: **Deploy from a branch**
- Branch: **main** / フォルダ: **/docs** → Save

数分後、レポートが下記URLで見られます:
```
https://<あなたのユーザー名>.github.io/stock-bot/
```

### 5. 動作テスト（手動実行）
- リポジトリの `Actions` タブ
- 左の **Daily ranking** → `Run workflow` ボタン
- 1〜2分で完了 → Pagesにレポートが反映

---

## これで完了

| | |
|---|---|
| 実行タイミング | 毎日16:00(JST)・平日・自動 |
| Mac | **不要**（GitHubのサーバーで動く） |
| 見る場所 | `https://<ユーザー名>.github.io/stock-bot/`（スマホでもOK） |
| 公開されるもの | コード・ランキング履歴のみ |
| 公開されないもの | 保有銘柄・ポートフォリオ・パスワード |

---

## 補足

- **ポートフォリオ（投信の資産）はクラウド化していません**。これまで通り
  Macで `python -m bot holdings report` を実行して `data/portfolio.html` を
  ローカルで見てください（資産情報を公開しないため）。
- ローカルの毎日自動実行（launchd）はそのまま残してもOKですが、クラウドが
  動けばランキングはそちらが正本になります。重複が気になれば launchd を
  止められます（言ってください）。
- GitHubの無料枠: public リポジトリの Actions は実質無制限。スケジュール実行は
  混雑時に数分遅れることがあります。
