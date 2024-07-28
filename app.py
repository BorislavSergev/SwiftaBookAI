from flask import Flask, jsonify
from training import get_machine_stats

app = Flask(__name__)

@app.route('/machine-stats', methods=['GET'])
def machine_stats():
    stats, status_code = get_machine_stats()
    return jsonify(stats), status_code

if __name__ == "__main__":
    app.run(port=5050, debug=True)
