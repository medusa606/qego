from enum import Enum

from gym.envs.classic_control import rendering

from library import mods
from library.bodies import TrafficLightState, Car, Bus, Bicycle, Pedestrian, PelicanCrossing
from library.geometry import Point


class RGB(Enum):
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 1)
    GREEN = (0, 1, 0)
    RED = (1, 0, 0)
    CYAN = (0, 1, 1)
    MAGENTA = (1, 0, 1)
    YELLOW = (1, 1, 0)
    WHITE = (1, 1, 1)


class BulbState(Enum):
    OFF = 0
    DIM = 1
    FULL = 2


class OcclusionView:
    def __init__(self, occlusion, ego, **kwargs):
        self.occlusion_zone = None

        if occlusion is not ego:
            self.occlusion_zone = rendering.make_polygon(list(occlusion.occlusion_zone(ego.state.position)), filled=False)
            self.occlusion_zone.set_color(*RGB.RED.value)

        super().__init__(**kwargs)  # important to pass on kwargs if class is used as superclass in multiple inheritance

    def update_occlusion_zone(self, occlusion, ego):
        if occlusion is not ego:
            self.occlusion_zone.v = list(occlusion.occlusion_zone(ego.state.position))


class BodyView:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # important to pass on kwargs if class is used as superclass in multiple inheritance

    def update(self, body, ego):
        raise NotImplementedError


def make_markers(points, radius=3):
    return rendering.Compound([mods.make_circle(*point, 3) for point in points])


class DynamicBodyView(BodyView, OcclusionView):
    def __init__(self, dynamic_body, ego, road):
        super().__init__(occlusion=dynamic_body, ego=ego)

        self.body = rendering.make_polygon(list(dynamic_body.bounding_box()))
        if dynamic_body is ego:
            self.body.set_color(*RGB.RED.value)

        self.focal_road = road
        if dynamic_body.bounding_box().intersects(self.focal_road.bounding_box()):
            self.road_angle = rendering.make_polyline(list())
        else:
            self.road_angle = rendering.make_polyline(list(dynamic_body.line_anchor(self.focal_road)))
        self.road_angle.set_color(*RGB.MAGENTA.value)

        if dynamic_body.target_spline:
            self.target_spline = rendering.make_polyline([tuple(point) for point in dynamic_body.target_spline])
            self.target_spline.set_color(*RGB.GREEN.value)
        else:
            self.target_spline = None

        if dynamic_body.planner_spline:
            self.planner_spline = rendering.make_polyline([tuple(point) for point in dynamic_body.planner_spline])
            self.planner_spline_markers = make_markers(dynamic_body.planner_spline)
        else:
            self.planner_spline = rendering.make_polyline(list())
            self.planner_spline_markers = make_markers(list())
        self.planner_spline.set_color(*RGB.BLUE.value)
        self.planner_spline_markers.set_color(*RGB.BLUE.value)

    def update(self, dynamic_body, ego):
        self.body.v = list(dynamic_body.bounding_box())

        self.update_occlusion_zone(dynamic_body, ego)

        if dynamic_body.bounding_box().intersects(self.focal_road.bounding_box()):
            self.road_angle.v = list()
        else:
            self.road_angle.v = list(dynamic_body.line_anchor(self.focal_road))

        if dynamic_body.planner_spline:
            self.planner_spline.v = [tuple(point) for point in dynamic_body.planner_spline]
            self.planner_spline_markers.gs = make_markers(dynamic_body.planner_spline).gs
        else:
            self.planner_spline.v = list()
            self.planner_spline_markers.gs = list()

    def geoms(self):
        if self.target_spline is not None:
            yield self.target_spline
        yield from [self.planner_spline, self.planner_spline_markers, self.body, self.road_angle]
        if self.occlusion_zone is not None:
            yield self.occlusion_zone


class VehicleView(DynamicBodyView):
    def __init__(self, vehicle, ego, road):
        super().__init__(vehicle, ego, road)

        braking_zone, reaction_zone = vehicle.stopping_zones()
        self.braking = rendering.make_polygon(list(braking_zone) if braking_zone else list(), filled=False)
        self.braking.set_color(*RGB.GREEN.value)
        self.braking.add_attr(mods.FactoredLineStyle(0x0F0F, 1))
        self.reaction = rendering.make_polygon(list(reaction_zone) if reaction_zone else list(), filled=False)
        self.reaction.set_color(*RGB.GREEN.value)

        lidar = vehicle.lidars()
        # self.lidar = rendering.make_polygon(list(lidar) if lidar else list())
        self.lidars = [rendering.make_polygon(list(lidar) if lidar else list()) for lidar in vehicle.lidars()]
        # self.lidar.set_color(*RGB.GREEN.value)
        # self.lidar.set_color(*RGB.YELLOW.value)
        # self.lidar.set_color(*RGB.RED.value)

        self.scale = {
            BulbState.OFF: 0.0,
            BulbState.DIM: vehicle.constants.width * 0.1,
            BulbState.FULL: vehicle.constants.width * 0.15
        }

        indicator_bounding_box = vehicle.indicators()
        self.left_indicators = self.make_lights(indicator_bounding_box.rear_left, indicator_bounding_box.front_left, BulbState.FULL)
        self.left_indicators.set_color(1, 0.75, 0)
        self.right_indicators = self.make_lights(indicator_bounding_box.rear_right, indicator_bounding_box.front_right, BulbState.FULL)
        self.right_indicators.set_color(1, 0.75, 0)

        longitudinal_bounding_box = vehicle.longitudinal_lights()
        self.brake_lights = self.make_lights(longitudinal_bounding_box.rear_left, longitudinal_bounding_box.rear_right, BulbState.FULL)
        self.brake_lights.set_color(*RGB.RED.value)
        self.headlights = self.make_lights(longitudinal_bounding_box.front_left, longitudinal_bounding_box.front_right, BulbState.FULL)
        self.headlights.set_color(*RGB.YELLOW.value)

    def make_lights(self, position1, position2, state):
        return rendering.Compound([
            mods.make_circle(*position1, self.scale[state]),
            mods.make_circle(*position2, self.scale[state])
        ])

    def update(self, vehicle, ego):
        super().update(vehicle, ego)

        # lidar = vehicle.lidar()
        # self.lidar.v = list(lidar) if lidar else list()
        for render_lidar, vehicle_lidar in zip(self.lidars, vehicle.lidars()):
            render_lidar.v = list(vehicle_lidar) if vehicle_lidar else list()

        braking_zone, reaction_zone = vehicle.stopping_zones()
        self.braking.v = list(braking_zone) if braking_zone else list()
        self.reaction.v = list(reaction_zone) if reaction_zone else list()

        left_indicator_state = BulbState.OFF
        right_indicator_state = BulbState.OFF
        if vehicle.steering_angle is not None:
            if vehicle.steering_angle > 0:
                left_indicator_state = BulbState.FULL
            elif vehicle.steering_angle < 0:
                right_indicator_state = BulbState.FULL

        indicator_bounding_box = vehicle.indicators()
        self.left_indicators.gs = self.make_lights(indicator_bounding_box.rear_left, indicator_bounding_box.front_left, left_indicator_state).gs
        self.right_indicators.gs = self.make_lights(indicator_bounding_box.rear_right, indicator_bounding_box.front_right, right_indicator_state).gs

        brake_lights_state = BulbState.OFF
        headlights_state = BulbState.OFF
        if vehicle.throttle is not None:
            if vehicle.throttle < 0:
                brake_lights_state = BulbState.FULL
            elif vehicle.throttle > 0:
                headlights_state = BulbState.FULL

        longitudinal_bounding_box = vehicle.longitudinal_lights()
        self.brake_lights.gs = self.make_lights(longitudinal_bounding_box.rear_left, longitudinal_bounding_box.rear_right, brake_lights_state).gs
        self.headlights.gs = self.make_lights(longitudinal_bounding_box.front_left, longitudinal_bounding_box.front_right, headlights_state).gs

    def geoms(self):
        yield from [self.braking, self.reaction]
        yield from super().geoms()
        yield from self.lidars


class CarView(VehicleView):
    def __init__(self, car, ego, road):
        super().__init__(car, ego, road)

        self.roof = rendering.make_polygon(list(car.roof()))
        if car is ego:
            self.roof.set_color(0.5, 0, 0)
        else:
            self.roof.set_color(0.5, 0.5, 0.5)

    def update(self, car, ego):
        super().update(car, ego)

        self.roof.v = list(car.roof())


def make_head(body):
    head = mods.make_circle(*body.state.position, body.constants.width * 0.3)
    head.set_color(0.5, 0.5, 0.5)
    return head


class BicycleView(DynamicBodyView):
    def __init__(self, bicycle, ego, road):
        super().__init__(bicycle, ego, road)

        self.head = make_head(bicycle)

    def update(self, bicycle, ego):
        super().update(bicycle, ego)

        self.head.v = make_head(bicycle).v


class PedestrianView(DynamicBodyView):
    def __init__(self, pedestrian, ego, road):
        super().__init__(pedestrian, ego, road)

        self.head = make_head(pedestrian)

    def update(self, pedestrian, ego):
        super().update(pedestrian, ego)

        self.head.v = make_head(pedestrian).v


class TrafficLightView(BodyView, OcclusionView):
    def __init__(self, traffic_light, ego):
        super().__init__(occlusion=traffic_light, ego=ego)

        self.body = rendering.make_polygon(list(traffic_light.static_bounding_box))

        self.red_light = mods.make_circle(*traffic_light.red_light, traffic_light.constants.width * 0.25)
        self.amber_light = mods.make_circle(*traffic_light.amber_light, traffic_light.constants.width * 0.25)
        self.green_light = mods.make_circle(*traffic_light.green_light, traffic_light.constants.width * 0.25)

        self.set_green_light()

    def set_red_light(self):
        self.red_light.set_color(*RGB.RED.value)
        for light in [self.amber_light, self.green_light]:
            light.set_color(*RGB.BLACK.value)

    def set_amber_light(self):
        self.amber_light.set_color(1, 0.75, 0)
        for light in [self.red_light, self.green_light]:
            light.set_color(*RGB.BLACK.value)

    def set_green_light(self):
        self.green_light.set_color(*RGB.GREEN.value)
        for light in [self.red_light, self.amber_light]:
            light.set_color(*RGB.BLACK.value)

    def update(self, traffic_light, ego):
        if traffic_light.state is TrafficLightState.RED:
            self.set_red_light()
        elif traffic_light.state is TrafficLightState.AMBER:
            self.set_amber_light()
        elif traffic_light.state is TrafficLightState.GREEN:
            self.set_green_light()

        self.update_occlusion_zone(traffic_light, ego)

    def geoms(self):
        yield from [self.body, self.red_light, self.amber_light, self.green_light, self.occlusion_zone]


class PelicanCrossingView(BodyView):
    def __init__(self, pelican_crossing, ego, **kwargs):
        super().__init__(**kwargs)

        coordinates = pelican_crossing.bounding_box()
        self.area = rendering.make_polygon(list(coordinates))
        self.area.set_color(*RGB.WHITE.value)

        self.markings = rendering.Compound([
            rendering.make_polyline([tuple(pelican_crossing.outbound_intersection_bounding_box.rear_left), tuple(pelican_crossing.outbound_intersection_bounding_box.rear_right)]),
            rendering.make_polyline([tuple(pelican_crossing.inbound_intersection_bounding_box.front_left), tuple(pelican_crossing.inbound_intersection_bounding_box.front_right)]),
            rendering.make_polyline([tuple(pelican_crossing.static_bounding_box.rear_left), tuple(pelican_crossing.static_bounding_box.front_left)]),
            rendering.make_polyline([tuple(pelican_crossing.static_bounding_box.rear_right), tuple(pelican_crossing.static_bounding_box.front_right)])
        ])

        offset_rear_right = Point(coordinates.rear_right.x + (pelican_crossing.constants.width * 0.15), coordinates.rear_right.y)
        offset_rear_left = Point(coordinates.rear_left.x + (pelican_crossing.constants.width * 0.15), coordinates.rear_left.y)
        offset_front_right = Point(coordinates.front_right.x - (pelican_crossing.constants.width * 0.15), coordinates.front_right.y)
        offset_front_left = Point(coordinates.front_left.x - (pelican_crossing.constants.width * 0.15), coordinates.front_left.y)
        self.inner = rendering.Compound([
            rendering.make_polyline([tuple(offset_rear_right), tuple(offset_rear_left)]),
            rendering.make_polyline([tuple(offset_front_right), tuple(offset_front_left)])
        ])
        self.inner.add_attr(mods.FactoredLineStyle(0x0F0F, 1))

        self.outbound_traffic_light_view = TrafficLightView(pelican_crossing.outbound_traffic_light, ego)
        self.inbound_traffic_light_view = TrafficLightView(pelican_crossing.inbound_traffic_light, ego)

    def update(self, pelican_crossing, ego):
        for traffic_light, traffic_light_view in zip([pelican_crossing.outbound_traffic_light, pelican_crossing.inbound_traffic_light], [self.outbound_traffic_light_view, self.inbound_traffic_light_view]):
            traffic_light_view.update(traffic_light, ego)

    def geoms(self):
        yield from [self.area, self.markings, self.inner]
        yield from self.outbound_traffic_light_view.geoms()
        yield from self.inbound_traffic_light_view.geoms()


class RoadView:
    def __init__(self, road):
        self.area = rendering.make_polygon(list(road.static_bounding_box))
        self.area.set_color(*RGB.WHITE.value)

        coordinates = road.bounding_box()
        self.edge_markings = rendering.Compound([
            rendering.make_polyline([tuple(coordinates.rear_left), tuple(coordinates.front_left)]),
            rendering.make_polyline([tuple(coordinates.rear_right), tuple(coordinates.front_right)])
        ])

        outbound_coordinates = road.outbound.bounding_box()
        self.centre_markings = rendering.make_polyline([tuple(outbound_coordinates.rear_right), tuple(outbound_coordinates.front_right)])
        self.centre_markings.add_attr(mods.FactoredLineStyle(0x00FF, 2))

        lane_lines = list()
        for lane in road.outbound.lanes[:-1] + road.inbound.lanes[:-1]:
            lane_coordinates = lane.bounding_box()
            lane_line = rendering.make_polyline([tuple(lane_coordinates.rear_right), tuple(lane_coordinates.front_right)])
            lane_lines.append(lane_line)
        self.lane_markings = rendering.Compound(lane_lines)
        self.lane_markings.add_attr(mods.FactoredLineStyle(0x00FF, 2))

        self.pelican_crossing_view = None

        self.bus_stop_views = list()
        for direction in [road.outbound, road.inbound]:
            if direction.bus_stop is not None:
                markings = rendering.make_polygon(list(direction.bus_stop.static_bounding_box), filled=False)
                self.bus_stop_views.append(markings)

    def set_pelican_crossing(self, pelican_crossing, ego):
        self.pelican_crossing_view = PelicanCrossingView(pelican_crossing, ego)

    def geoms(self):
        yield from [self.area, self.centre_markings, self.lane_markings, self.edge_markings]
        if self.bus_stop_views is not None:
            yield from self.bus_stop_views


class ObstacleView(OcclusionView):
    def __init__(self, obstacle, ego):
        super().__init__(obstacle, ego)

        self.body = rendering.make_polygon(list(obstacle.static_bounding_box))

    def geoms(self):
        yield from [self.body, self.occlusion_zone]

    def update(self, obstacle, ego):
        self.update_occlusion_zone(obstacle, ego)


class RoadMapView:
    def __init__(self, road_map, body):
        self.major_road_view = RoadView(road_map.major_road)
        self.minor_road_views = [RoadView(minor_road) for minor_road in road_map.minor_roads] if road_map.minor_roads is not None else list()

        if self.minor_road_views:
            self.clear_intersections = rendering.Compound([rendering.make_polyline([tuple(bounding_box.front_left), tuple(bounding_box.front_right)]) for bounding_box in road_map.intersection_bounding_boxes])
            self.clear_intersections.set_color(*RGB.WHITE.value)

            self.intersection_markings = rendering.Compound([rendering.make_polyline([tuple(bounding_box.rear_left), tuple(bounding_box.rear_right)]) for bounding_box in road_map.inbound_intersection_bounding_boxes])
            self.intersection_markings.add_attr(mods.FactoredLineStyle(0x0F0F, 2))

        self.obstacle_view = None
        if road_map.obstacle is not None:
            self.obstacle_view = ObstacleView(road_map.obstacle, body)

    def geoms(self):
        for minor_road_view in self.minor_road_views:
            yield from minor_road_view.geoms()
        yield from self.major_road_view.geoms()
        if self.minor_road_views:
            yield from [self.clear_intersections, self.intersection_markings]
        if self.obstacle_view:
            yield from self.obstacle_view.geoms()


class RoadEnvViewer(rendering.Viewer):
    def __init__(self, width, height, road_map, bodies, ego):
        super().__init__(width=int(width), height=int(height))  # width and height must be integers
        self.road_map = road_map

        self.transform.set_translation(0.0, self.height / 2.0)  # Specify that (0, 0) should be centre-left of viewer (default is bottom-left)

        self.road_map_view = RoadMapView(road_map, ego)

        for geom in self.road_map_view.geoms():
            self.add_geom(geom)

        self.body_views = list()
        for body in bodies:
            if isinstance(body, Car):
                car_view = CarView(body, ego, road_map.major_road)
                self.body_views.append(car_view)
            elif isinstance(body, Bus):
                bus_view = VehicleView(body, ego, road_map.major_road)
                self.body_views.append(bus_view)
            elif isinstance(body, Bicycle):
                bicycle_view = BicycleView(body, ego, road_map.major_road)
                self.body_views.append(bicycle_view)
            elif isinstance(body, Pedestrian):
                pedestrian_view = PedestrianView(body, ego, road_map.major_road)
                self.body_views.append(pedestrian_view)
            elif isinstance(body, PelicanCrossing):
                self.road_map_view.major_road_view.set_pelican_crossing(body, ego)
                self.body_views.append(self.road_map_view.major_road_view.pelican_crossing_view)

        if self.road_map_view.major_road_view.pelican_crossing_view is not None:
            for geom in self.road_map_view.major_road_view.pelican_crossing_view.geoms():
                self.add_geom(geom)

        self.dynamic_body_views = [body_view for body_view in self.body_views if isinstance(body_view, DynamicBodyView)]

        for dynamic_body_view in self.dynamic_body_views:
            for geom in dynamic_body_view.geoms():
                self.add_geom(geom)

        self.vehicle_views = [dynamic_body_view for dynamic_body_view in self.dynamic_body_views if isinstance(dynamic_body_view, VehicleView)]

        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.right_indicators)
            self.add_geom(vehicle_view.left_indicators)

        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.brake_lights)

        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.headlights)

        for dynamic_body_view in self.dynamic_body_views:
            self.add_geom(dynamic_body_view.body)

        self.car_views = [vehicle_view for vehicle_view in self.vehicle_views if isinstance(vehicle_view, CarView)]

        for car_view in self.car_views:
            self.add_geom(car_view.roof)

        self.bicycle_views = [dynamic_body_view for dynamic_body_view in self.dynamic_body_views if isinstance(dynamic_body_view, BicycleView)]

        for bicycle_view in self.bicycle_views:
            self.add_geom(bicycle_view.head)

        self.pedestrian_views = [dynamic_body_view for dynamic_body_view in self.dynamic_body_views if isinstance(dynamic_body_view, PedestrianView)]

        for pedestrian_view in self.pedestrian_views:
            self.add_geom(pedestrian_view.head)

    def update(self, bodies, ego):
        for body_view, body in zip(self.body_views, bodies):
            body_view.update(body, ego)

        if self.road_map_view.obstacle_view is not None:
            self.road_map_view.obstacle_view.update(self.road_map.obstacle, ego)
