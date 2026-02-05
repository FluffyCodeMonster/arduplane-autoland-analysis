# Utility functions

import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import art3d, proj3d
from matplotlib.patches import ArrowStyle
from matplotlib.patches import FancyArrowPatch
from ardupilot_log_reader import Ardupilot
from pymavlink import mavutil
from geographiclib.geodesic import Geodesic as geodesic_gglib
from scipy.interpolate import make_interp_spline

@np.vectorize
def _ned_to_xyz_vectorised(n, e, d):
    return e, n, -d


def ned_to_xyz(n, e, d):
    # Call vectorised method
    transformed_coords = _ned_to_xyz_vectorised(n, e, d)
    return np.array(transformed_coords)


def xyz_to_ned(x, y, z):
    # In the current setup, can just reapply the ned_to_xyz transformation to invert.
    return ned_to_xyz(x, y, z)


@np.vectorize(signature='(),(),()->(3,3)')
def calc_body_to_inertial(phi, theta, psi):
    return np.array([
        [np.cos(theta)*np.cos(psi),   np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),   np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
        [np.cos(theta)*np.sin(psi),   np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi),   np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
        [-np.sin(theta),               np.sin(phi)*np.cos(theta),                                            np.cos(phi)*np.cos(theta)]
        ])


class Plane:
    # '_xyz' indicates that the position coordinates are in the xyz frame
    def __init__(self, ax, scale=1, init_x_xyz=0, init_y_xyz=0, init_z_xyz=0, init_phi=0, init_theta=0, init_psi=0, colour=None, linewidth=None, zorder=1e5):
        # https://stackoverflow.com/questions/4622057/plotting-3d-polygons
        # WOT4: wingspan 1.33m, length 1.2m
        v = scale*np.array([
            [1.5, 0, 0],
            [-1, 1, 0],
            [-0.5, 0, -0.3],
            [-1, -1, 0],
        ])
        f = [[0, 1, 2], [0, 2, 3]]
        # Done like this so that the code in set_pose can still rotate the vertices.
        self.verts = np.array([[v[i] for i in p] for p in f])
        self.verts_flat = self.verts.reshape(np.prod(self.verts.shape[:2]), 3)
        self.plane = art3d.Poly3DCollection(self.verts)
        self.plane.set_zorder(zorder)
        # Each plane will be a random colour if not specified
        self.plane.set_color(colour if colour else colors.rgb2hex(np.random.rand(3)))
        self.plane.set_edgecolor('k')
        if linewidth is not None:
            self.plane.set_linewidth(linewidth)

        ax.add_collection3d(self.plane)

        # Set initial position and orientation
        self.set_pose(init_x_xyz, init_y_xyz, init_z_xyz, init_phi, init_theta, init_psi)
    
    def transform_verts(self, verts):
        return verts.reshape(*self.verts.shape[:2], 3)

    def set_pose(self, posnx_xyz, posny_xyz, posnz_xyz, phi, theta, psi):
        # Orient
        ## Body to inertial matrix calculates the inertial frame coordinates of a vector expressed in the body
        ## frame. Alternatively, it can be used to rotate a vector in the fixed inertial perspective by the
        ## orientation of the body frame. This is what it is being used for here - to align the aircraft polygon
        ## with the body frame.
        # Using this matrix, we can treat the aircraft vertices as if they are in the body frame, and then find their inertial frame positions for plotting.
        rotat = calc_body_to_inertial(phi, theta, psi)
        # verts needs to be in the format...
        #  +-                    -+
        #  |   |    |    |        |
        #  |  pt1  pt2  pt3  ...  |
        #  |   |    |    |        |
        #  +-                    -+
        # so that each vertex point is rotated by the rotation matrix. This is why self.verts
        # is transposed, since this is the opposite orientation to that required by Poly3DCollection.
        oriented_verts = np.matmul(rotat, self.verts_flat.T)

        # Convert NED system coordinates (for *plane vertices* - position of plane already given in xyz coord system) to
        # xyz coord system used by Matplotlib for plotting.
        # (Need to flip E and D)
        # [TODO Is this true, following code modifications?] NOTE The oriented_verts array is changed by the ned_to_xyz() function, so it technically doesn't have to be returned and re-assigned, but it helps to make
        # the code more maintainable by making value changes more obvious.
        # TODO [Do a proper comment for this] Want to use the scalar version so that it comes out as a 2D Numpy array. Note that the expansion operator (*)
        # has to be used in the argument.
        oriented_verts = ned_to_xyz(*oriented_verts)

        # Translate
        # posn_x_xyz, posn_y_xyz and posn_z_xyz should be in the xyz coordinate system used by Matplotlib for plotting.
        shifted_verts = oriented_verts + np.array([[posnx_xyz], [posny_xyz], [posnz_xyz]])

        # .T to transpose back to the expected format.
        self.plane.set_verts(self.transform_verts(shifted_verts.T))


# Fix to make 3D quiver plot work. Code from: https://github.com/matplotlib/matplotlib/issues/21688.
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    # FT modification - 3/11/23
    def update_3d_posns(self, xs, ys, zs):
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


# Can be used as a base class or on its own
# colours, line_styles: [list]s of colour and line_style strings. Both should contain the same number of elements.
# plot_inds: None to use all
class VecPlotterBase:
    def __init__(self, ax, colours, line_styles=None, plot_inds=None, scaler_func=lambda x: x, head_width=0.2, line_width=1, alpha=1, zorder=0):
        # Create list of vectors based on given colours
        self.vec_arrows = []
        if not line_styles:
            line_styles = ['-']*len(colours)
        for c, ls in zip(colours, line_styles):
            # CurveFilledB = "-|>"
            vec_arrow = Arrow3D([0, 0], [0, 0], [0, 0], mutation_scale=10, lw=line_width, arrowstyle=ArrowStyle.CurveFilledB(head_width=head_width), color=c, linestyle=ls, alpha=alpha)
            vec_arrow.set_zorder(zorder)
            self.vec_arrows.append(vec_arrow)
            ax.add_artist(vec_arrow)
        
        self.plot_inds = plot_inds
        self.scaler_func = scaler_func
    
    # Separated out in case there are multiple vectors to iterate through, as there are in the case of the forces.
    # vecs is a Numpy array of vector lengths - shape = (n, 3).
    def update(self, n, e, d, vecs):
        for i in range(len(vecs)):
            if (self.plot_inds is None) or (i in self.plot_inds):
                vec_data = vecs[i]
                vec_arrow = self.vec_arrows[i]

                # Hide arrows when the force is 0
                if np.linalg.norm(vec_data) == 0:
                    vec_arrow.set(visible=False)
                else:
                    vec_arrow.set(visible=True)

                ### ned -> xyz conversion has already been performed for x, y and z coordinates
                x, y, z = ned_to_xyz(n, e, d)
                vec_xyz = ned_to_xyz(*vec_data)
                vec_x_xyz, vec_y_xyz, vec_z_xyz = vec_xyz

                # Transform to spherical coordinates...
                # theta = np.arctan(vec_y_xyz / vec_x_xyz)
                theta = np.arctan2(vec_y_xyz, vec_x_xyz)
                # phi = np.arctan(vec_z_xyz / np.sqrt(vec_y_xyz**2 + vec_x_xyz**2))
                phi = np.arctan2(vec_z_xyz, np.sqrt(vec_y_xyz**2 + vec_x_xyz**2))
                
                # ...scale...
                # Scale on the norm of the vector, so that the x, y and z components are all scaled by the same amount.
                vec_len = self.scaler_func(np.linalg.norm(vec_xyz))

                # ...then transform back
                # sin(phi) = O/H
                vec_z_xyz_scaled = vec_len*np.sin(phi)
                vec_y_xyz_scaled = (vec_len*np.cos(phi))*np.sin(theta)
                vec_x_xyz_scaled = (vec_len*np.cos(phi))*np.cos(theta)

                vec_arrow.update_3d_posns([x, x + vec_x_xyz_scaled], [y, y + vec_y_xyz_scaled], [z, z + vec_z_xyz_scaled])


class VecPlotterSingle(VecPlotterBase):
    def __init__(self, ax, colour, **kwargs):
        super().__init__(ax, [colour], **kwargs)

    # Call for single vector vec
    def update(self, n, e, d, vec):
        super().update(n, e, d, np.array([vec]))


# Need to calculate these distances for each waypoint
# Altitude not actually required for this
@np.vectorize
def calc_rel_pos_ned(init_lat, init_lon, init_alt, lat, lon, alt):
    lat_dist_m = geodesic_gglib.WGS84.Inverse(init_lat, init_lon, lat, init_lon)['s12']*np.sign(lat - init_lat)
    lon_dist_m = geodesic_gglib.WGS84.Inverse(init_lat, init_lon, init_lat, lon)['s12']*np.sign(lon - init_lon)
    return lat_dist_m, lon_dist_m, -(alt - init_alt)


# Load the log file
class LogFileLoader:
    def __init__(self, path, types=['STAT', 'XKF1', 'ARSP', 'AOA', 'ATT', 'IMU', 'AETR', 'TECS', 'AHR2', 'RFND', 'POS', 'BARO', 'GPS', 'LAND']):
        self.log_file_path = path
        self.name = path.name
        self.types = types

        self.populate()

        # Download waypoints
        self.download_waypoints()

        # Get glight mode information
        self.get_flight_mode_info()

        # For accessors
        self.n_values = None
        self.e_values = None
        self.d_values = None
    
    def get_flight_logs(self):
        print("Parsing flight logs... ", end="")
        parser = Ardupilot.parse(self.log_file_path, types=self.types)
        print("done")
        
        return parser

    # f() is a function, for if it's necessary. E.g. to get altitude, have to multiply PD by -1. Might also
    # want to convert radians to degrees, etc.
    def construct_interpolator(self, logs, log_type, key, k=1, f=lambda x: x):
        # unique_times, unique_idx = np.unique(np.array(logs.dfs[log_type]['timestamp']), return_index=True)
        unique_times, unique_idx = np.unique(np.array(logs.dfs[log_type]['TimeUS']) / 1e6, return_index=True)
        interp_vals = np.array(logs.dfs[log_type][key])[unique_idx]
        interp = make_interp_spline(unique_times, f(interp_vals), k=k)
        return interp
    
    def populate(self):
        self.logs = self.get_flight_logs()
        
        # Get the start and end times from STAT
        # This is in seconds - time since system startup
        stat_times = np.array(self.logs.dfs['STAT']['TimeUS']) / 1e6 # ['timestamp'])
        self.start_time_s = stat_times[0]
        self.end_time_s = stat_times[-1]

        self.time_values_s = np.arange(self.start_time_s, self.end_time_s, 0.05)
        self.normalised_time_values_s = self.time_values_s - self.time_values_s[0]

        # print(times_s)
        print(f"Normalised start time (s): {self.normalised_time_values_s[0]}, normalised end time (s): {self.normalised_time_values_s[-1]}")

        # ===================================================
        # Interpolate in case the different logs are taken at different times

        # Positions (metres)
        #self.ns = self.construct_interpolator(logs, 'XKF1', 'PN')
        #self.es = self.construct_interpolator(logs, 'XKF1', 'PE')
        #self.ds = self.construct_interpolator(logs, 'XKF1', 'PD')

        # Air-relative velocity
        self.vas_interp = self.construct_interpolator(self.logs, 'ARSP', 'Airspeed')
        # Angles of attack and sideslip (degrees)
        self.alphas_interp = self.construct_interpolator(self.logs, 'AOA', 'AOA')
        self.betas_interp = self.construct_interpolator(self.logs, 'AOA', 'SSA')
        
        # Calculate groundspeed values
        # TODO - calculate from air-relative velocity values and angle transformation matrix

        # Euler angles (degrees)
        self.phis_interp = self.construct_interpolator(self.logs, 'ATT', 'Roll')
        self.thetas_interp = self.construct_interpolator(self.logs, 'ATT', 'Pitch')
        self.psis_interp = self.construct_interpolator(self.logs, 'ATT', 'Yaw')
        
        # Angular rates
        self.ps_interp = self.construct_interpolator(self.logs, 'IMU', 'GyrX', f=lambda x: np.degrees(x))
        self.qs_interp = self.construct_interpolator(self.logs, 'IMU', 'GyrY', f=lambda x: np.degrees(x))
        self.rs_interp = self.construct_interpolator(self.logs, 'IMU', 'GyrZ', f=lambda x: np.degrees(x))
        
        # Control values
        self.ctrl_as_interp = self.construct_interpolator(self.logs, 'AETR', 'Ail')
        self.ctrl_es_interp = self.construct_interpolator(self.logs, 'AETR', 'Elev')
        self.ctrl_ts_interp = self.construct_interpolator(self.logs, 'AETR', 'Thr')
        self.ctrl_rs_interp = self.construct_interpolator(self.logs, 'AETR', 'Rudd')
        self.ctrl_flaps_interp = self.construct_interpolator(self.logs, 'AETR', 'Flap')
        
        # Desired values
        self.phis_des_interp = self.construct_interpolator(self.logs, 'ATT', 'DesRoll')
        self.thetas_des_interp = self.construct_interpolator(self.logs, 'ATT', 'DesPitch')
        self.psis_des_interp = self.construct_interpolator(self.logs, 'ATT', 'DesYaw')
        
        # TECS values
        self.tecs_h_interp = self.construct_interpolator(self.logs, 'TECS', 'h')
        self.tecs_hdem_interp = self.construct_interpolator(self.logs, 'TECS', 'hdem')
        self.tecs_dh_interp = self.construct_interpolator(self.logs, 'TECS', 'dh')
        self.tecs_dhdem_interp = self.construct_interpolator(self.logs, 'TECS', 'dhdem')
        self.tecs_sp_interp = self.construct_interpolator(self.logs, 'TECS', 'sp')
        self.tecs_spdem_interp = self.construct_interpolator(self.logs, 'TECS', 'spdem')
        self.tecs_dsp_interp = self.construct_interpolator(self.logs, 'TECS', 'dsp')
        self.tecs_dspdem_interp = self.construct_interpolator(self.logs, 'TECS', 'dspdem')
        self.tecs_th_interp = self.construct_interpolator(self.logs, 'TECS', 'th')
        self.tecs_ph_interp = self.construct_interpolator(self.logs, 'TECS', 'ph')
        self.tecs_ph_interp = self.construct_interpolator(self.logs, 'TECS', 'ph')

        # Landing parameters
        # TODO I don't know what all of these parameters are/do
        self.land_stage_interp = self.construct_interpolator(self.logs, 'LAND', 'stage')
        self.land_f1_interp = self.construct_interpolator(self.logs, 'LAND', 'f1')
        self.land_f2_interp = self.construct_interpolator(self.logs, 'LAND', 'f2')
        self.land_slope_interp = self.construct_interpolator(self.logs, 'LAND', 'slope')
        self.land_slopeInit_interp = self.construct_interpolator(self.logs, 'LAND', 'slopeInit')
        self.land_altO_interp = self.construct_interpolator(self.logs, 'LAND', 'altO')
        self.land_fh_interp = self.construct_interpolator(self.logs, 'LAND', 'fh')

        # Latitude and longitude (degrees)
        self.lat_degs_pos_interp = self.construct_interpolator(self.logs, 'POS', 'Lat') # , k=1)  # TODO Why did I use k=1 here?
        self.lon_degs_pos_interp = self.construct_interpolator(self.logs, 'POS', 'Lng') # , k=1)  # TODO Why did I use k=1 here?
        self.alt_ms_pos_interp = self.construct_interpolator(self.logs, 'POS', 'Alt') # , k=1)  # TODO Why did I use k=1 here?
        
        self.lat_degs_ahr2_interp = self.construct_interpolator(self.logs, 'AHR2', 'Lat') # , k=1)  # TODO Why did I use k=1 here?
        self.lon_degs_ahr2_interp = self.construct_interpolator(self.logs, 'AHR2', 'Lng') # , k=1)  # TODO Why did I use k=1 here?
        self.alt_ms_ahr2_interp = self.construct_interpolator(self.logs, 'AHR2', 'Alt') # , k=1)  # TODO Why did I use k=1 here?
        
        # Barometric height
        self.alt_ms_baro_amsl_interp = self.construct_interpolator(self.logs, 'BARO', 'AltAMSL')
        # GPS height
        self.alt_ms_gps_interp = self.construct_interpolator(self.logs, 'GPS', 'Alt')
        
        # Rangefinder distance (metres)
        self.rfnd_dist_ms_interp = self.construct_interpolator(self.logs, 'RFND', 'Dist')

        self.xkf1_vn_interp = self.construct_interpolator(self.logs, 'XKF1', 'VN')
        self.xkf1_ve_interp = self.construct_interpolator(self.logs, 'XKF1', 'VE')
        self.xkf1_vd_interp = self.construct_interpolator(self.logs, 'XKF1', 'VD')
    
    def set_inds(self, min_ind, max_ind):
        self.min_ind = min_ind
        self.max_ind = max_ind
    
    def calc_neds(self):
        self.n_values, self.e_values, self.d_values = calc_rel_pos_ned(
            self.lat_degs_pos_interp(self.time_values_s[0]),
            self.lon_degs_pos_interp(self.time_values_s[0]),
            self.alt_ms_pos_interp(self.time_values_s[0]),
            self.lat_degs_pos_interp(self.time_values_s),
            self.lon_degs_pos_interp(self.time_values_s),
            self.alt_ms_pos_interp(self.time_values_s))
    
    # Return pos altitude at start_time_s
    def get_start_pos_alt_ms(self):
        return self.alt_ms_pos_interp(self.start_time_s)
    
    # Return rangefinder distance at start_time_s
    def get_start_rfnd_dist_ms(self):
        return self.rfnd_dist_ms_interp(self.start_time_s)
    
    def get_param(self, param_name):
        # return self.logs.dfs['PARM'][param_name]
        return self.logs.dfs['PARM'][self.logs.dfs['PARM']['Name'] == param_name]['Value'].to_numpy().item()
    
    @property
    def times_s(self):
        return self.time_values_s[self.min_ind:self.max_ind]
    
    @property
    def normalised_times_s(self):
        return self.normalised_time_values_s[self.min_ind:self.max_ind]
    
    @property
    def ns(self):
        if self.n_values is None:
            self.calc_neds()
        return self.n_values[self.min_ind:self.max_ind]
    
    @property
    def es(self):
        if self.e_values is None:
            self.calc_neds()
        return self.e_values[self.min_ind:self.max_ind]
    
    @property
    def ds(self):
        if self.d_values is None:
            self.calc_neds()
        return self.d_values[self.min_ind:self.max_ind]
    
    @property
    def vas(self):
        return self.vas_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    # Euler angles (degrees)
    @property
    def phis(self):
        return self.phis_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def thetas(self):
        return self.thetas_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def psis(self):
        return self.psis_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    # Control values
    @property
    def ctrl_as(self):
        return self.ctrl_as_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def ctrl_es(self):
        return self.ctrl_es_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def ctrl_ts(self):
        return self.ctrl_ts_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def ctrl_rs(self):
        return self.ctrl_rs_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def ctrl_flaps(self):
        return self.ctrl_flaps_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    # Desired values
    @property
    def phis_des(self):
        return self.phis_des_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def thetas_des(self):
        return self.thetas_des_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def psis_des(self):
        return self.psis_des_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    # ===

    @property
    def tecs_h(self):
        return self.tecs_h_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_hdem(self):
        return self.tecs_hdem_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_dh(self):
        return self.tecs_dh_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_dhdem(self):
        return self.tecs_dhdem_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_sp(self):
        return self.tecs_sp_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_spdem(self):
        return self.tecs_spdem_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_dsp(self):
        return self.tecs_dsp_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_dspdem(self):
        return self.tecs_dspdem_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_th(self):
        return self.tecs_th_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_ph(self):
        return self.tecs_ph_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def tecs_ph(self):
        return self.tecs_ph_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    # ===
    
    @property
    def land_stage(self):
        return self.land_stage_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def land_f1(self):
        return self.land_f1_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def land_f2(self):
        return self.land_f2_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def land_slope(self):
        return self.land_slope_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def land_slopeInit(self):
        return self.land_slopeInit_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def land_altO(self):
        return self.land_altO_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def land_fh(self):
        return self.land_fh_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def alt_ms_pos(self):
        return self.alt_ms_pos_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def alt_ms_ahr2(self):
        return self.alt_ms_ahr2_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def alt_ms_baro_amsl(self):
        return self.alt_ms_baro_amsl_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def alt_ms_gps(self):
        return self.alt_ms_gps_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def rfnd_dist_ms(self):
        return self.rfnd_dist_ms_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def vn_xkf1_m(self):
        return self.xkf1_vn_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def ve_xkf1_m(self):
        return self.xkf1_ve_interp(self.time_values_s)[self.min_ind:self.max_ind]
    
    @property
    def vd_xkf1_m(self):
        return self.xkf1_vd_interp(self.time_values_s)[self.min_ind:self.max_ind]
        
    def download_waypoints(self):
        # Connect to log file
        conn = mavutil.mavlink_connection(str(self.log_file_path))

        self.wps = []
        self.wp_frames = []
        self.wps_ned = []
        first_seen = False  # Ignore the first waypoint - it is in a different frame
        while (cmd_msg := conn.recv_match(type='CMD', blocking=True)) is not None:
            if not first_seen:
                first_seen = True
                continue
            self.wps.append((cmd_msg.Lat, cmd_msg.Lng, cmd_msg.Alt))
            self.wp_frames.append(cmd_msg.Frame)
            # self.wps_ned.append(calc_rel_pos_ned(self.lat_degs_pos_interp(self.start_time_s), self.lon_degs_pos_interp(self.start_time_s), self.alt_ms_pos_interp(self.start_time_s), cmd_msg.Lat, cmd_msg.Lng, cmd_msg.Alt))
            self.wps_ned.append(calc_rel_pos_ned(self.lat_degs_pos_interp(self.start_time_s), self.lon_degs_pos_interp(self.start_time_s), 0, cmd_msg.Lat, cmd_msg.Lng, cmd_msg.Alt))
        
        conn.close()
    
    def get_flight_mode_info(self):
        # Connect to log file
        conn = mavutil.mavlink_connection(str(self.log_file_path))

        self.mode_times_normalised_s = []
        self.modes = []
        while (mode_msg := conn.recv_match(type='MODE', blocking=True)) is not None:
            # Need to convert to normalised seconds
            self.mode_times_normalised_s.append((mode_msg.TimeUS / 1e6) - self.start_time_s)
            self.modes.append(mode_msg.Mode)
            # print("Under construction")
        
        conn.close()