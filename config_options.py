'''
####################
TESTER CONFIGS
####################
  "tester_config": {
    "option": "random",
    "epsilon": 0.01 },

  "tester_config": {
    "option": "random-constrained",
    "epsilon": 0.01 },

"tester_config": {
    "option": "proximity",
    "threshold": 544.0},

"tester_config": {
    "option": "election",
    "threshold": 544.0},

"tester_config": {
    "option": "q-learning",
    "alpha":{
      "start": 0.18,
      "stop": 0.18 ,
      "num_steps": 10000 },
    "gamma": 0.87  ,
    "epsilon": 0.0005,
    "feature_config": {
      "distance_x": true,
      "distance_y": true,
      "distance": true,
      "relative_angle": false,
      "heading": false,
      "on_road": false,
      "inverse_distance": true
    },


####################
EGO CONFIGS
####################


  "ego_config": {
    "option": "noop",
    "log": null },

 config
  "ego_config": {
    "option": "q-learning",
    "alpha":{
      "start": 0.0008,
      "stop": 0.00008 ,
      "num_steps": 10000 },
    "gamma": 0.5  ,
    "epsilon": 0.1,
    "feature_config": {
      "distance_x": false,
      "distance_y": false,
      "distance": false,
      "relative_angle": true,
      "heading": false,
      "on_road": false,
      "inverse_distance": false
    },
    "log": null },

'''
