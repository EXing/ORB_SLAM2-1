//
// Created by huangkun on 19-7-18.
//

#include <spline/SplineBA.hpp>
#include <string>
#include <set>
#include<opencv2/core/eigen.hpp>

inline void spline::SplineBA::rotMatrixToDer(const Eigen::Matrix3d &Rbw, Eigen::Vector3d &d, double &theta) {
    d = Rbw.row(1).transpose();

    Eigen::Vector3d n = d.cross(Eigen::Vector3d(0, 0, 1));
    n /= n.norm();
    Eigen::Vector3d z = n.cross(d);

    double tmp = Rbw.row(2) * z;
    theta = std::acos(tmp > 1.0 ? 1.0 : tmp);// [0, pi]

    if (std::acos(Rbw.row(2) * n) > M_PI / 2)
        theta = -theta;
}

inline void spline::SplineBA::derToRotMatrix(const Eigen::Vector3d &d, double theta, Eigen::Matrix3d &Rbw) {
    Eigen::Vector3d unit_d = d / d.norm();
    Rbw.row(1) = unit_d.transpose();
    Eigen::Vector3d n = unit_d.cross(Eigen::Vector3d(0, 0, 1));
    n /= n.norm();
    Eigen::Vector3d z = n.cross(unit_d);
    z /= z.norm();

    Rbw.row(2) = (std::cos(theta) * z + std::sin(theta) * n).transpose();
    Rbw.row(0) = Rbw.row(1).cross(Rbw.row(2)); // X = Y x Z
    Rbw.row(0) /= Rbw.row(0).norm();
}

spline::SplineBA::SplineBA(std::vector<KeyFrame *> &vpKF, const std::vector<MapPoint *> &vpMP,
                           const bool bRobust) {
    sort(vpKF.begin(), vpKF.end(), KeyFrame::lId);

    /*** Construction ***/
    Eigen::Matrix3d Rcb;
    Eigen::Vector3d tcb;
    Rcb << 1, 0, 0, 0, 0, -1, 0, 1, 0;
    tcb << 0, 0, 0;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Q, dQ;
    std::vector<double> u;
    int counter = 0;
    for (auto &it: vpKF) {
        if (it->isBad())
            continue;

        double theta;
        Eigen::Vector3d d;
        cv::Mat cvTwc = it->GetPoseInverse();
        Eigen::Matrix3d Rwb, Rwc;
        Rwc << cvTwc.at<float>(0, 0), cvTwc.at<float>(0, 1), cvTwc.at<float>(0, 2),
                cvTwc.at<float>(1, 0), cvTwc.at<float>(1, 1), cvTwc.at<float>(1, 2),
                cvTwc.at<float>(2, 0), cvTwc.at<float>(2, 1), cvTwc.at<float>(2, 2);
        Rwb = Rwc * Rcb;
        Eigen::Vector3d twc(cvTwc.at<float>(0, 3), cvTwc.at<float>(1, 3), cvTwc.at<float>(2, 3));
        Eigen::Vector3d twb = Rwc * tcb + twc;
        rotMatrixToDer(Rwb.transpose(), d, theta);

        Q.emplace_back(twb);
        dQ.push_back(d);
        rotAngle.push_back(theta);
        u.push_back(it->mTimeStamp);
        KFidLook[it->mnId] = counter++;
    }

    std::vector<double> init_rotAngle(rotAngle);// backup

    // Approximation
    traj = spline::Bspline<3>(3, Q, std::floor(vpKF.size() / 3), u, dQ);

    std::vector<double> splineInlierRate;
    optimize(vpKF, vpMP, true, splineInlierRate, bRobust);

    /*std::set<size_t> X;
    for (size_t i = 0; i < splineInlierRate.size(); i++) {
        if (splineInlierRate[i] < 0.7) {// i-th frame
            size_t idx = traj.findSpan(u[i]);
            X.insert(idx);
        }
    }
    std::vector<double> Xv;
    for (auto &idx: X) {
        Xv.push_back((traj.getKnotVector()[idx] + traj.getKnotVector()[idx + 1]) / 2);
    }
    traj.refineKnotVect(Xv);
    optimize(vpKF, vpMP, false, splineInlierRate, bRobust);*/
    /*// TODO: is it reasonable to run opt several times? any other choice?
    for (int counter = 0; counter < 5; counter++) {// 5 iteration at most
        if (std::accumulate(splineInlierRate.begin(), splineInlierRate.end(), 0.0) / splineInlierRate.size() < 0.5) {
            // reinitialization
            rotAngle = init_rotAngle;
            traj = spline::Bspline<3>(3, Q, floor(vpKF.size() / (2 - 0.1 * counter)), u, dQ);
            optimize(vpKF, vpMP, true, splineInlierRate, bRobust);
        } else { // knot refinement
            std::set<double> X;
            for (size_t i = 0; i < splineInlierRate.size(); i++) {
                if (splineInlierRate[i] < 0.5) {// i-th frame
                    auto idx = traj.findSpan(u[i]);
                    X.insert((traj.getKnotVector()[idx] + traj.getKnotVector()[idx + 1]) / 2);
                }
            }
            if (X.empty()) {
                break;
            }
            std::vector<double> Xv(X.begin(), X.end());
            traj.refineKnotVect(Xv);
            optimize(vpKF, vpMP, false, splineInlierRate, bRobust);
        }
    }*/

    // Store opt result
    for (auto &it: vpKF) { // keyframe
        if (it->isBad())
            continue;

        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Ders;
        traj.evaluate(it->mTimeStamp, 1, Ders);
        Eigen::Matrix3d Rbw;
        derToRotMatrix(Ders[1], rotAngle[KFidLook[it->mnId]], Rbw);
        Eigen::Matrix4d Tcw;
        Eigen::Vector3d twb = Ders[0];
        Eigen::Vector3d tbw = -Rbw * twb;
        Tcw.block<3, 3>(0, 0) = Rcb * Rbw;
        Tcw.block<3, 1>(0, 3) = Rcb * tbw + tcb;

        Eigen::Matrix4f Tcwf = Tcw.cast<float>();
        cv::eigen2cv(Tcwf, it->mTcwGBA);
    }

    for (size_t i = 0; i < vpMP.size(); i++) {
        MapPoint *pMP = vpMP[i];
        if (pMP->isBad())
            continue;

        pMP->mPosGBA.convertTo(pMP->mPosGBA, CV_32F);
    }
}

/*** optimized result stored in:  rotAngle, traj.CP, pMP->mPosGBA ***/
int spline::SplineBA::optimize(const std::vector<KeyFrame *> &vpKF, const std::vector<MapPoint *> &vpMP,
                               bool initialization, std::vector<double> &splineInlierRate, const bool bRobust) {
    /*** Optimize control points, rotAng and landmarks w.r.t reprojection error ***/
    ceres::Problem problem;
    ceres::Solver::Options options;
    auto *ordering = new ceres::ParameterBlockOrdering;
    std::unordered_map<unsigned long, std::vector<ceres::ResidualBlockId>> splineResidualBlocks;

    std::vector<ceres::ResidualBlockId> reprojectionRes, rotConsistencyRes, rotPenaltyRes;

    // Rcb, tcb
    Eigen::Matrix3d Rcb;
    Eigen::Vector3d tcb;
    Rcb << 1, 0, 0, 0, 0, -1, 0, 1, 0;
    tcb << 0, 0, 0;

    // spline derivatives
    std::unordered_map<unsigned long, Eigen::Matrix<double, 2, 4>, hash<unsigned long>, std::equal_to<unsigned long>, Eigen::aligned_allocator<std::allocator<std::pair<const unsigned long, Eigen::Matrix<double, 2, 4>>>>> ders_ptrs;
    std::unordered_map<unsigned long, size_t> spanIdxs;
    for (auto &it: vpKF) {
        if (it->isBad())
            continue;

        size_t spanIdx = traj.findSpan(it->mTimeStamp);
        std::vector<std::vector<double>> ders;
        traj.dersBasisFuns(it->mTimeStamp, spanIdx, 1, ders);
        Eigen::Matrix<double, 2, 4> ders_ptr;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                ders_ptr(i, j) = ders[i][j];
            }
        }
        ders_ptrs[it->mnId] = ders_ptr;
        spanIdxs[it->mnId] = spanIdx;
    }

    // SplineReprojectionError
    const double thHuber2D = std::sqrt(5.99);
    const double thHuber3D = std::sqrt(7.815);
    ceres::LossFunction *loss_function_2d = new ceres::HuberLoss(thHuber2D);
    ceres::LossFunction *loss_function_3d = new ceres::HuberLoss(thHuber3D);
    for (size_t i = 0; i < vpMP.size(); i++) {
        MapPoint *pMP = vpMP[i];
        if (pMP->isBad())
            continue;
        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {
            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > vpKF.back()->mnId)
                continue;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];
            if (pKF->mvuRight[mit->second] < 0) { // Mono
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ceres::CostFunction *cost_function = SplineMonoReprojectionError::Create(obs, Rcb, tcb, pKF->fx,
                                                                                         pKF->fy, pKF->cx, pKF->cy,
                                                                                         ders_ptrs[pKF->mnId]);


                if (initialization) {
                    // Remind convert CV_32F to CV_64F for double type usage
                    pMP->GetWorldPos().convertTo(pMP->mPosGBA, CV_64F);
                }

                size_t spanIdx = spanIdxs[pKF->mnId];
                reprojectionRes.push_back(
                        problem.AddResidualBlock(cost_function, bRobust ? loss_function_2d : nullptr,
                                                 traj.getCP()[spanIdx - 3 + 0].data(),
                                                 traj.getCP()[spanIdx - 3 + 1].data(),
                                                 traj.getCP()[spanIdx - 3 + 2].data(),
                                                 traj.getCP()[spanIdx - 3 + 3].data(),
                                                 rotAngle.data() + KFidLook[pKF->mnId], pMP->mPosGBA.ptr<double>()));
                splineResidualBlocks[pKF->mnId].push_back(reprojectionRes.back());

                // ordering, landmarks 1st
                ordering->AddElementToGroup(pMP->mPosGBA.ptr<double>(), 0);
                //problem.SetParameterBlockConstant(pMP->mPosGBA.ptr<double>());
            } else { // Stereo
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                ceres::CostFunction *cost_function = SplineStereoReprojectionError::Create(obs, Rcb, tcb, pKF->fx,
                                                                                           pKF->fy, pKF->cx, pKF->cy,
                                                                                           pKF->mbf,
                                                                                           ders_ptrs[pKF->mnId]);

                if (initialization) {
                    pMP->GetWorldPos().convertTo(pMP->mPosGBA, CV_64F);
                }

                size_t spanIdx = spanIdxs[pKF->mnId];
                reprojectionRes.push_back(
                        problem.AddResidualBlock(cost_function, bRobust ? loss_function_3d : nullptr,
                                                 traj.getCP()[spanIdx - 3 + 0].data(),
                                                 traj.getCP()[spanIdx - 3 + 1].data(),
                                                 traj.getCP()[spanIdx - 3 + 2].data(),
                                                 traj.getCP()[spanIdx - 3 + 3].data(),
                                                 rotAngle.data() + KFidLook[pKF->mnId], pMP->mPosGBA.ptr<double>()));
                splineResidualBlocks[pKF->mnId].push_back(reprojectionRes.back());

                // ordering, landmarks 1st
                ordering->AddElementToGroup(pMP->mPosGBA.ptr<double>(), 0);
                //problem.SetParameterBlockConstant(pMP->mPosGBA.ptr<double>());
            }
        }
    }

    /*// RotConsistencyError
    for (size_t i = 1; i < rotAngle.size(); i++) {
        rotConsistencyRes.push_back(
                problem.AddResidualBlock(RotConsistencyError::Create(), nullptr, rotAngle.data() + i,
                                         rotAngle.data() + i - 1));
    }*/

    for (size_t i = 0; i < rotAngle.size(); i++) {
        // RotPenalty
        rotPenaltyRes.push_back(problem.AddResidualBlock(RotPenalty::Create(), nullptr, rotAngle.data() + i));

        // RotAng constrain
        problem.SetParameterLowerBound(rotAngle.data() + i, 0, -M_PI / 4);
        problem.SetParameterUpperBound(rotAngle.data() + i, 0, M_PI / 4);

        // ordering, rotAngle 2nd
        ordering->AddElementToGroup(rotAngle.data() + i, 1);
    }

    // ordering, control points 3rd
    for (auto &ref: traj.getCP()) {
        ordering->AddElementToGroup(ref.data(), 2);
    }

    // Fix 1st Control point
    problem.SetParameterBlockConstant(traj.getCP()[0].data());

    // Scale Consistency except buffer begin(TODO: windowBA)
    // Fix the first p+1 control points
    problem.SetParameterBlockConstant(traj.getCP()[1].data());
    problem.SetParameterBlockConstant(traj.getCP()[2].data());
    problem.SetParameterBlockConstant(traj.getCP()[3].data());

    /*** error before opt ***/
    ceres::Problem::EvaluateOptions evaluateOptions;
    std::vector<double> residuals;

    evaluateOptions.residual_blocks = reprojectionRes;
    problem.Evaluate(evaluateOptions, nullptr, &residuals, nullptr, nullptr);
    double repErrorBefore = 0;
    for (auto &res: residuals) {
        repErrorBefore += res * res;
    }
    residuals.clear();

    /*evaluateOptions.residual_blocks = rotConsistencyRes;
    problem.Evaluate(evaluateOptions, nullptr, &residuals, nullptr, nullptr);
    double rotConsistencyErrorBefore = 0;
    for (auto &res: residuals) {
        rotConsistencyErrorBefore += res * res;
    }
    residuals.clear();*/

    evaluateOptions.residual_blocks = rotPenaltyRes;
    problem.Evaluate(evaluateOptions, nullptr, &residuals, nullptr, nullptr);
    double rotPenaltyErrorBefore = 0;
    for (auto &res: residuals) {
        rotPenaltyErrorBefore += res * res;
    }
    residuals.clear();

    /*** solve ***/
    options.linear_solver_ordering.reset(ordering);
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 3;
    //options.max_num_iterations = 20;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Use FullReport() to diagnose performance problems
    std::cout << summary.FullReport() << "\n";

    /*if (initialization) {
        // count inliers
        splineInlierRate.clear();
        double inlierThreshold = 2; // TODO: Find a suitable inlierThreshold in sqrt pixel unit
        for (auto &it: vpKF) {
            if (it->isBad())
                continue;
            evaluateOptions.residual_blocks = std::move(splineResidualBlocks[it->mnId]);
            std::vector<double> singleFrameResiduals;
            problem.Evaluate(evaluateOptions, nullptr, &singleFrameResiduals, nullptr, nullptr);
            splineInlierRate.push_back(std::count_if(singleFrameResiduals.begin(), singleFrameResiduals.end(),
                                                     [&](double &x) { return std::abs(x) < inlierThreshold; })
                                       / double(singleFrameResiduals.size()));
        }
    }*/

    /*** error after opt ***/
    evaluateOptions.residual_blocks = reprojectionRes;
    problem.Evaluate(evaluateOptions, nullptr, &residuals, nullptr, nullptr);
    double repErrorAfter = 0;
    for (auto &res: residuals) {
        repErrorAfter += res * res;
    }
    residuals.clear();

    /*evaluateOptions.residual_blocks = rotConsistencyRes;
    problem.Evaluate(evaluateOptions, nullptr, &residuals, nullptr, nullptr);
    double rotConsistencyErrorAfter = 0;
    for (auto &res: residuals) {
        rotConsistencyErrorAfter += res * res;
    }
    residuals.clear();*/

    evaluateOptions.residual_blocks = rotPenaltyRes;
    problem.Evaluate(evaluateOptions, nullptr, &residuals, nullptr, nullptr);
    double rotPenaltyErrorAfter = 0;
    for (auto &res: residuals) {
        rotPenaltyErrorAfter += res * res;
    }
    residuals.clear();

    std::cout << "Error before opt:" << std::endl
              << "repError: " << repErrorBefore << std::endl
              //<< "rotConsistencyError: " << rotConsistencyErrorBefore << std::endl
              << "rotPenaltyError: " << rotPenaltyErrorBefore << std::endl
              << "Error after opt:" << std::endl
              << "repError: " << repErrorAfter << std::endl
              //<< "rotConsistencyError: " << rotConsistencyErrorAfter << std::endl
              << "rotPenaltyErrorError: " << rotPenaltyErrorAfter << std::endl;

    return 0;
}