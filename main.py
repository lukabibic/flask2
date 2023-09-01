import numpy as np
from flask import Flask, jsonify, request, send_from_directory
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
from flask import render_template
import logging
from matplotlib import pyplot as plt
import seaborn as sns
from flask import send_file
import os
import matplotlib.colors as mcolors

if not os.path.exists('static'):
    os.makedirs('static')

if not os.path.exists('static/output_folder'):
    os.makedirs('static/output_folder')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

logging.basicConfig(level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

G = ox.graph_from_place('Rijeka, Croatia', network_type='drive', simplify=False)  # simplify False gives more nodes

matrix_image_path = os.path.join(os.getcwd(), 'static', 'output_folder', 'matrix_image.png')

if os.path.exists(matrix_image_path):
    app.logger.debug("deleting existing file")
    os.remove(matrix_image_path)


def calculate_le_lt(point1, point2):
    node1 = ox.distance.nearest_nodes(G, point1[1], point1[0])
    node2 = ox.distance.nearest_nodes(G, point2[1], point2[0])

    road_distance = nx.shortest_path_length(G, node1, node2, weight='length')
    air_distance = geodesic(point1, point2).meters

    if road_distance < 1e-6:
        return float(-1)  # return -1 to indicate undefined

    return air_distance / road_distance


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
def home():
    return render_template('map.html')


def calculate(point1, point2):
    app.logger.debug(f"point1: {point1}, point2: {point2}")

    # Get the nearest nodes to the clicked points
    node1 = ox.distance.nearest_nodes(G, point1[1], point1[0])
    node2 = ox.distance.nearest_nodes(G, point2[1], point2[0])
    app.logger.debug(f"node1: {node1}, node2: {node2}")

    # Use NetworkX to calculate the shortest path
    shortest_path_nodes = nx.shortest_path(G, node1, node2)
    app.logger.debug(f"shortest_path_nodes: {shortest_path_nodes}")

    # Create the shortest path line coordinates
    shortest_path_line = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in shortest_path_nodes]
    app.logger.debug(f"shortest_path_line: {shortest_path_line}")

    # Calculate the road distance
    road_distance = nx.shortest_path_length(G, node1, node2, weight='length')
    app.logger.debug(f"road_distance: {road_distance}")

    # Calculate the air distance
    air_distance = geodesic(point1, point2).meters
    app.logger.debug(f"air_distance: {air_distance}")

    # Calculate the Le/Lt factor
    le_lt_factor = air_distance / road_distance
    app.logger.debug(f'The Le/Lt factor is {le_lt_factor}.')

    return air_distance, road_distance, shortest_path_line, le_lt_factor


@app.route('/get_matrix_image')
def get_matrix_image():
    # return send_file(matrix_image_path, mimetype='image/png')
    app.logger.debug("Matrix image requested")
    return send_from_directory("static/output_folder", "matrix_image.png")


@app.route('/calculate_le_lt_matrix', methods=['POST'])
def calculate_le_lt_matrix():
    data = request.get_json()
    points = data['points']

    n = len(points)
    matrix = np.zeros((n, n))
    paths = []

    for i in range(n):
        for j in range(n):
            if i != j:
                # Using the calculate function to get air and road distances and le_lt factor
                air_distance, road_distance, road_path, le_lt = calculate(points[i], points[j])
                matrix[i][j] = le_lt

                paths.append({
                    "start": i,
                    "end": j,
                    "path": road_path,
                    "air_distance": air_distance,
                    "road_distance": road_distance
                })

    colors = ["red", "orange", "green", "blue"]
    boundaries = [0, 0.25, 0.5, 0.75, 1]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    sns.heatmap(matrix, annot=True, cmap=cmap, norm=norm)

    # plt.show()

    # Save the image
    if os.path.exists(matrix_image_path):
        app.logger.debug("deleting existing file")
        os.remove(matrix_image_path)

    plt.savefig(matrix_image_path)

    # Close the plot
    plt.close()

    app.logger.debug(f"calculated matrix: {matrix}")

    return jsonify({
        'matrix': matrix.tolist(),
        'paths': paths,
    })


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))