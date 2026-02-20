const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");

module.exports = {
  entry: "./src/index.js",
  output: {
    filename: "bundle.[contenthash].js",
    path: path.resolve(__dirname, "dist"),
    clean: true,
  },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: "./src/index.html",
      title: "Zelda OoS RL Map Visualizer",
    }),
    new CopyWebpackPlugin({
      patterns: [{ from: "assets", to: "assets" }],
    }),
  ],
  devServer: {
    static: {
      directory: path.join(__dirname, "assets"),
      publicPath: "/assets",
    },
    port: 8080,
    hot: true,
  },
};
