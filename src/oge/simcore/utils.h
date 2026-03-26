//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_UTILS_H
#define ORBITALGAMEENV_UTILS_H

#include <cmath>
#include <chrono>

#include "oge/common/constants.h"
#include "oge/simcore/propagator.h"

namespace oge
{
    /**
     * This function evaluates the Stumpff function S(z) according to Eq 3.49.
     * @param z input argument
     * @return value of S(z)
     */
    double stumpS(double z);

    /**
     * This function evaluates the Stumpff function C(z) according to Eq 3.50.
     * @param z input argument
     * @return value of C(z)
     */
    double stumpC(double z);

    /**
     *
     * @param [input] x     the universal anomaly after time t (kmˆ0.5)
     * @param [input] t     the time elapsed since t (s)
     * @param [input] ro    the radial position at time t (km)
     * @param [input] a     reciprocal of the semimajor axis (1/km)
     * @param [output] f    the Lagrange f coefficient (dimensionless)
     * @param [output] g    the Lagrange g coefficient (s)
     */
    void f_and_g(double x, double t, double ro, double a, double& f, double& g);

    void fDot_and_gDot(double x, double r, double ro, double a, double& fdot, double& gdot);

    /**
     * This function computes the state vector (r,v) from the classical orbital elements (coe).
     * @param[in] coe orbital elements [a, e, incl, RA, w, TA]
     * - a semimajor axis (km)
     * - e eccentricity
     * - incl inclination of the orbit (rad)
     * - RA right ascension of the ascending node (rad)
     * - w argument of perigee (rad)
     * - TA true anomaly (rad)
     * @param[out] R position vector in the geocentric equatorial frame (km)
     * @param[out] V velocity vector in the geocentric equatorial frame (km/s)
     */
    void coe2rv(const Eigen::Matrix<double, 6, 1>& coe, Eigen::Vector3d& R, Eigen::Vector3d& V);

    /**
     * This function computes the classical orbital elements (coe)
     * from the state vector (R,V) using Algorithm 4.1.
     * @param R position vector in the geocentric equatorial frame (km)
     * @param V velocity vector in the geocentric equatorial frame (km/s)
     * @param coe vector of orbital elements [a, e, incl, RA, w, TA]
     * - a semimajor axis (km)
     * - e eccentricity
     * - incl inclination of the orbit (rad)
     * - RA right ascension of the ascending node (rad)
     * - w argument of perigee (rad)
     * - TA true anomaly (rad))
     */
    void rv2coe(const Eigen::Vector3d& R, const Eigen::Vector3d& V, Eigen::Matrix<double, 6, 1>& coe);

    double rad2deg(double rad);
    double deg2rad(double deg);

    void DCM_LVLH_to_J2000(const Eigen::Vector3d& RRefJ2000, const Eigen::Vector3d& VRefJ2000, Eigen::Matrix3d& DCM);
    void DCM_J2000_to_LVLH(const Eigen::Vector3d& RRefJ2000, const Eigen::Vector3d& VRefJ2000, Eigen::Matrix3d& DCM);

    /**
     * This function transforms a position and velocity vector from the LVLH frame
     * to the J2000 inertial frame, given a reference orbit in J2000.
     * The LVLH frame is defined with x-axis along the radial direction,
     * z-axis along the orbit normal, and y-axis completing the right-handed system
     * (approximately along the velocity direction).
     * @param[in] RRefJ2000 position vector of the reference orbit in the J2000 frame (km)
     * @param[in] VRefJ2000 velocity vector of the reference orbit in the J2000 frame (km/s)
     * @param[in] R_LVLH position vector in the LVLH frame (km)
     * @param[in] V_LVLH velocity vector in the LVLH frame (km/s)
     * @param[out] R_J2000 position vector in the J2000 frame (km)
     * @param[out] V_J2000 velocity vector in the J2000 frame (km/s)
     */
    void RV_LVLH2J2000(
        const Eigen::Vector3d& RRefJ2000,
        const Eigen::Vector3d& VRefJ2000,
        const Eigen::Vector3d& R_LVLH,
        const Eigen::Vector3d& V_LVLH,
        Eigen::Vector3d& R_J2000,
        Eigen::Vector3d& V_J2000);

    /**
     * This function transforms a position and velocity vector from the J2000 inertial frame
     * to the LVLH frame, given a reference orbit in J2000.
     * The LVLH frame is defined with x-axis along the radial direction,
     * z-axis along the orbit normal, and y-axis completing the right-handed system
     * (approximately along the velocity direction).
     * @param[in] RRefJ2000 position vector of the reference orbit in the J2000 frame (km)
     * @param[in] VRefJ2000 velocity vector of the reference orbit in the J2000 frame (km/s)
     * @param[in] R_J2000 position vector in the J2000 frame (km)
     * @param[in] V_J2000 velocity vector in the J2000 frame (km/s)
     * @param[out] R_LVLH position vector in the LVLH frame (km)
     * @param[out] V_LVLH velocity vector in the LVLH frame (km/s)
     */
    void RV_J20002LVLH(
        const Eigen::Vector3d& RRefJ2000,
        const Eigen::Vector3d& VRefJ2000,
        const Eigen::Vector3d& R_J2000,
        const Eigen::Vector3d& V_J2000,
        Eigen::Vector3d& R_LVLH,
        Eigen::Vector3d& V_LVLH);

    //年月日时分秒转化为儒略日
    double JulianDay(std::chrono::sys_time<std::chrono::seconds> & UTC);

    //年月日时分秒转化为MJD儒略日
    double JulianDay_Modified(std::chrono::sys_time<std::chrono::seconds>& UTC);

    //UTC时间转换为TDT时间. 输入: UTC MJD格式的UTC.
    double UTC_TDT(double MJD);

    //TDT时间转换为MJD格式的UTC时间. 输入: TDT MJD格式的TDT.
    double TDT_UTC(double TDT);

    //MJD时间转换为UTC时间. 输入: MJD时间.
    void MJD_UTC(double MJD, std::chrono::sys_time<std::chrono::seconds>& UTC);

    // 计算太阳在 J2000 坐标系下的位置
    void solar_position(double jd, Eigen::Vector3d& pos_sun_j2000);
    void solar_position(const std::chrono::sys_time<std::chrono::seconds>& UTC, Eigen::Vector3d& pos_sun_j2000);
}

#endif //ORBITALGAMEENV_UTILS_H
