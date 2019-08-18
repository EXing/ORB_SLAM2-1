//
// Created by huangkun on 19-7-12.
//

#ifndef SPLINEBA_HPP
#define SPLINEBA_HPP

#include <spline/Bspline.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <mutex>
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"
#include <unordered_map>

using namespace ORB_SLAM2;

namespace spline { // Default: Assume car front is $z$-axis

    class SplineBA {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        int
        optimize(const std::vector<KeyFrame *> &vpKF, const std::vector<MapPoint *> &vpMP, bool initialization,
                 std::vector<double> &splineInlierRate, const bool bRobust = true);

        SplineBA(std::vector<KeyFrame *> &vpKF, const std::vector<MapPoint *> &vpMP, const bool bRobust = true);

        inline void rotMatrixToDer(const Eigen::Matrix3d &Rbw, Eigen::Vector3d &d, double &theta);

        inline void derToRotMatrix(const Eigen::Vector3d &d, double theta, Eigen::Matrix3d &Rbw);

    protected:
        spline::Bspline<3> traj;
        std::vector<double> rotAngle;
        std::unordered_map<unsigned long, int> KFidLook;
    };

    /*** template fucntion for ceres ***/
    template<typename T, int row_stride, int col_stride>
    inline
    void derToRotMatrix(
            const T *y, const T theta,
            const ceres::MatrixAdapter<T, row_stride, col_stride> &Rbw) {
        Rbw(1, 0) = y[0];
        Rbw(1, 1) = y[1];
        Rbw(1, 2) = y[2];

        const T g[3] = {T(0), T(0), T(1)};
        T n[3];
        ceres::CrossProduct<T>(y, g, n);
        const T scale = T(1) / sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        n[0] *= scale;
        n[1] *= scale;
        n[2] *= scale;

        T z[3];
        ceres::CrossProduct<T>(n, y, z);
        const T cost = ceres::cos(theta);
        const T sint = ceres::sin(theta);
        z[0] = cost * z[0] + sint * n[0];
        z[1] = cost * z[1] + sint * n[1];
        z[2] = cost * z[2] + sint * n[2];
        Rbw(2, 0) = z[0];
        Rbw(2, 1) = z[1];
        Rbw(2, 2) = z[2];

        T x[3];
        ceres::CrossProduct<T>(y, z, x);
        Rbw(0, 0) = x[0];
        Rbw(0, 1) = x[1];
        Rbw(0, 2) = x[2];
    }

    struct SplineMonoReprojectionError {
        SplineMonoReprojectionError(const Eigen::Vector2d &obs, const Eigen::Matrix3d &Rcb, const Eigen::Vector3d &tcb,
                                    double fx, double fy, double cx, double cy,
                                    Eigen::Matrix<double, 2, 4> &dersTraj)
                : obs(obs), Rcb(Rcb), tcb(tcb), dersTraj(dersTraj), fx(fx), fy(fy), cx(cx), cy(cy) {}

        template<typename T>
        bool operator()(const T *const cp0, const T *const cp1, const T *const cp2, const T *const cp3,
                        const T *const rotAng, const T *const lm,
                        T *residuals) const {
            const T twb[3] = {
                    T(dersTraj(0, 0)) * cp0[0] + T(dersTraj(0, 1)) * cp1[0] + T(dersTraj(0, 2)) * cp2[0] +
                    T(dersTraj(0, 3)) * cp3[0],
                    T(dersTraj(0, 0)) * cp0[1] + T(dersTraj(0, 1)) * cp1[1] + T(dersTraj(0, 2)) * cp2[1] +
                    T(dersTraj(0, 3)) * cp3[1],
                    T(dersTraj(0, 0)) * cp0[2] + T(dersTraj(0, 1)) * cp1[2] + T(dersTraj(0, 2)) * cp2[2] +
                    T(dersTraj(0, 3)) * cp3[2]};

            T y[3] = {
                    T(dersTraj(1, 0)) * cp0[0] + T(dersTraj(1, 1)) * cp1[0] + T(dersTraj(1, 2)) * cp2[0] +
                    T(dersTraj(1, 3)) * cp3[0],
                    T(dersTraj(1, 0)) * cp0[1] + T(dersTraj(1, 1)) * cp1[1] + T(dersTraj(1, 2)) * cp2[1] +
                    T(dersTraj(1, 3)) * cp3[1],
                    T(dersTraj(1, 0)) * cp0[2] + T(dersTraj(1, 1)) * cp1[2] + T(dersTraj(1, 2)) * cp2[2] +
                    T(dersTraj(1, 3)) * cp3[2]};
            const T scale = T(1) / ceres::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
            y[0] *= scale;
            y[1] *= scale;
            y[2] *= scale;

            T Rbw[9];
            const auto &Rbw_adapter = ceres::ColumnMajorAdapter3x3(Rbw);
            derToRotMatrix(y, rotAng[0], Rbw_adapter);

            // tbw = - Rbw * twb
            const T tbw[3] = {
                    -(Rbw_adapter(0, 0) * twb[0] + Rbw_adapter(0, 1) * twb[1] + Rbw_adapter(0, 2) * twb[2]),
                    -(Rbw_adapter(1, 0) * twb[0] + Rbw_adapter(1, 1) * twb[1] + Rbw_adapter(1, 2) * twb[2]),
                    -(Rbw_adapter(2, 0) * twb[0] + Rbw_adapter(2, 1) * twb[1] + Rbw_adapter(2, 2) * twb[2])};

            T Xb[3] = {Rbw_adapter(0, 0) * lm[0] + Rbw_adapter(0, 1) * lm[1] + Rbw_adapter(0, 2) * lm[2],
                       Rbw_adapter(1, 0) * lm[0] + Rbw_adapter(1, 1) * lm[1] + Rbw_adapter(1, 2) * lm[2],
                       Rbw_adapter(2, 0) * lm[0] + Rbw_adapter(2, 1) * lm[1] + Rbw_adapter(2, 2) * lm[2]};
            Xb[0] += tbw[0];
            Xb[1] += tbw[1];
            Xb[2] += tbw[2];

            T Xc[3] = {T(Rcb(0, 0)) * Xb[0] + T(Rcb(0, 1)) * Xb[1] + T(Rcb(0, 2)) * Xb[2],
                       T(Rcb(1, 0)) * Xb[0] + T(Rcb(1, 1)) * Xb[1] + T(Rcb(1, 2)) * Xb[2],
                       T(Rcb(2, 0)) * Xb[0] + T(Rcb(2, 1)) * Xb[1] + T(Rcb(2, 2)) * Xb[2]};
            Xc[0] += T(tcb(0));
            Xc[1] += T(tcb(1));
            Xc[2] += T(tcb(2));

            residuals[0] = T(obs[0]) - (Xc[0] / Xc[2] * T(fx) + T(cx));
            residuals[1] = T(obs[1]) - (Xc[1] / Xc[2] * T(fy) + T(cy));

            return true;
        }

        static ceres::CostFunction *
        Create(const Eigen::Vector2d &obs, const Eigen::Matrix3d &Rcb, const Eigen::Vector3d &tcb, double fx, double fy,
               double cx, double cy, Eigen::Matrix<double, 2, 4> &dersTraj) {
            return (new ceres::AutoDiffCostFunction<SplineMonoReprojectionError, 2, 3, 3, 3, 3, 1, 3>(
                    new SplineMonoReprojectionError(obs, Rcb, tcb, fx, fy, cx, cy, dersTraj)));
        }

        Eigen::Vector2d obs;
        Eigen::Matrix3d Rcb;
        Eigen::Vector3d tcb;
        double fx, fy, cx, cy;

        // for spline evaluation
        Eigen::Matrix<double, 2, 4> dersTraj;
    };

    struct SplineStereoReprojectionError {
        SplineStereoReprojectionError(const Eigen::Vector3d &obs, const Eigen::Matrix3d &Rcb,
                                      const Eigen::Vector3d &tcb,
                                      double fx, double fy, double cx, double cy, double bf,
                                      Eigen::Matrix<double, 2, 4> &dersTraj)
                : obs(obs), Rcb(Rcb), tcb(tcb), dersTraj(dersTraj), fx(fx), fy(fy), cx(cx), cy(cy), bf(bf) {}

        template<typename T>
        bool operator()(const T *const cp0, const T *const cp1, const T *const cp2, const T *const cp3,
                        const T *const rotAng, const T *const lm,
                        T *residuals) const {
            const T twb[3] = {
                    T(dersTraj(0, 0)) * cp0[0] + T(dersTraj(0, 1)) * cp1[0] + T(dersTraj(0, 2)) * cp2[0] +
                    T(dersTraj(0, 3)) * cp3[0],
                    T(dersTraj(0, 0)) * cp0[1] + T(dersTraj(0, 1)) * cp1[1] + T(dersTraj(0, 2)) * cp2[1] +
                    T(dersTraj(0, 3)) * cp3[1],
                    T(dersTraj(0, 0)) * cp0[2] + T(dersTraj(0, 1)) * cp1[2] + T(dersTraj(0, 2)) * cp2[2] +
                    T(dersTraj(0, 3)) * cp3[2]};

            T y[3] = {
                    T(dersTraj(1, 0)) * cp0[0] + T(dersTraj(1, 1)) * cp1[0] + T(dersTraj(1, 2)) * cp2[0] +
                    T(dersTraj(1, 3)) * cp3[0],
                    T(dersTraj(1, 0)) * cp0[1] + T(dersTraj(1, 1)) * cp1[1] + T(dersTraj(1, 2)) * cp2[1] +
                    T(dersTraj(1, 3)) * cp3[1],
                    T(dersTraj(1, 0)) * cp0[2] + T(dersTraj(1, 1)) * cp1[2] + T(dersTraj(1, 2)) * cp2[2] +
                    T(dersTraj(1, 3)) * cp3[2]};
            const T scale = T(1) / ceres::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
            y[0] *= scale;
            y[1] *= scale;
            y[2] *= scale;

            T Rbw[9];
            const auto &Rbw_adapter = ceres::ColumnMajorAdapter3x3(Rbw);
            derToRotMatrix(y, rotAng[0], Rbw_adapter);

            // tbw = - Rbw * twb
            const T tbw[3] = {
                    -(Rbw_adapter(0, 0) * twb[0] + Rbw_adapter(0, 1) * twb[1] + Rbw_adapter(0, 2) * twb[2]),
                    -(Rbw_adapter(1, 0) * twb[0] + Rbw_adapter(1, 1) * twb[1] + Rbw_adapter(1, 2) * twb[2]),
                    -(Rbw_adapter(2, 0) * twb[0] + Rbw_adapter(2, 1) * twb[1] + Rbw_adapter(2, 2) * twb[2])};

            T Xb[3] = {Rbw_adapter(0, 0) * lm[0] + Rbw_adapter(0, 1) * lm[1] + Rbw_adapter(0, 2) * lm[2],
                       Rbw_adapter(1, 0) * lm[0] + Rbw_adapter(1, 1) * lm[1] + Rbw_adapter(1, 2) * lm[2],
                       Rbw_adapter(2, 0) * lm[0] + Rbw_adapter(2, 1) * lm[1] + Rbw_adapter(2, 2) * lm[2]};
            Xb[0] += tbw[0];
            Xb[1] += tbw[1];
            Xb[2] += tbw[2];

            T Xc[3] = {T(Rcb(0, 0)) * Xb[0] + T(Rcb(0, 1)) * Xb[1] + T(Rcb(0, 2)) * Xb[2],
                       T(Rcb(1, 0)) * Xb[0] + T(Rcb(1, 1)) * Xb[1] + T(Rcb(1, 2)) * Xb[2],
                       T(Rcb(2, 0)) * Xb[0] + T(Rcb(2, 1)) * Xb[1] + T(Rcb(2, 2)) * Xb[2]};
            Xc[0] += T(tcb(0));
            Xc[1] += T(tcb(1));
            Xc[2] += T(tcb(2));

            T tmp = Xc[0] / Xc[2] * T(fx) + T(cx);

            residuals[0] = T(obs[0]) - tmp;
            residuals[1] = T(obs[1]) - (Xc[1] / Xc[2] * T(fy) + T(cy));
            residuals[2] = T(obs[2]) - (tmp - T(bf) / Xc[2]);

            return true;
        }

        static ceres::CostFunction *
        Create(const Eigen::Vector3d &obs, const Eigen::Matrix3d &Rcb, const Eigen::Vector3d &tcb, double fx, double fy,
               double cx, double cy, double bf, Eigen::Matrix<double, 2, 4> &dersTraj) {
            return (new ceres::AutoDiffCostFunction<SplineStereoReprojectionError, 3, 3, 3, 3, 3, 1, 3>(
                    new SplineStereoReprojectionError(obs, Rcb, tcb, fx, fy, cx, cy, bf, dersTraj)));
        }

        Eigen::Vector3d obs;
        Eigen::Matrix3d Rcb;
        Eigen::Vector3d tcb;
        double fx, fy, cx, cy, bf;

        // for spline evaluation
        Eigen::Matrix<double, 2, 4> dersTraj;
    };

    struct RotConsistencyError {
        template<typename T>
        bool operator()(const T *const rotAng0, const T *const rotAng1, T *residuals) const {
            residuals[0] = T(5e3) * (rotAng0[0] - rotAng1[0]);
            return true;
        }

        static ceres::CostFunction *
        Create() {
            return (new ceres::AutoDiffCostFunction<RotConsistencyError, 1, 1, 1>(new RotConsistencyError()));
        }
    };

    struct RotPenalty {
        template<typename T>
        bool operator()(const T *const rotAng, T *residuals) const {
            residuals[0] = T(1e2) * (rotAng[0]);
            return true;
        }

        static ceres::CostFunction *
        Create() {
            return (new ceres::AutoDiffCostFunction<RotPenalty, 1, 1>(new RotPenalty()));
        }
    };
}

#endif //SPLINEBA_HPP
