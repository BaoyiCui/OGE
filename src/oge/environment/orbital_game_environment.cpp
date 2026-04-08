//
// Created by baoyicui on 2/22/26.
//

#include "oge/simcore/math.h"
#include "oge/simcore/utils.h"

#include "orbital_game_environment.h"

namespace oge
{
    OrbitalGameEnvironment::OrbitalGameEnvironment(
        const OGESettings& settings_
    ) :
        settings(settings_),
        random_seed(settings_.getInt("random_seed", true)),
        // simulation settings
        dv_init_red(settings_.getFloat("dv_init_red")),
        dv_init_blue(settings_.getFloat("dv_init_blue")),
        dv_max_per_step_red(settings_.getFloat("dv_max_per_step_red")),
        dv_max_per_step_blue(settings_.getFloat("dv_max_per_step_blue")),
        capture_distance(settings_.getFloat("capture_distance")),
        timestep(settings_.getFloat("timestep")),
        terminal_time(settings_.getFloat("terminal_time")),
        // random initialization settings
        sma_perturb_max(settings_.getFloat("sma_perturb_max")),
        dist_init_offset_max(settings_.getFloat("dist_init_offset_max")),
        dist_init_offset_min(settings_.getFloat("dist_init_offset_min")),
        // reward settings
        reward_time_weight(settings_.getFloat("reward_time_weight")),
        reward_formation_weight(settings_.getFloat("reward_formation_weight")),
        reward_fuel_weight(settings_.getFloat("reward_fuel_weight")),
        reward_capture_weight(settings_.getFloat("reward_capture_weight")),
        reward_timeout_weight(settings_.getFloat("reward_timeout_weight")),
        reward_fuelout_weight(settings_.getFloat("reward_fuelout_weight")),
        reward_phase_dist_weight(settings_.getFloat("reward_phase_dist_weight")),
        // distance reward sub-parameters
        reward_far_sma_penalty_scale(settings_.getFloat("reward_far_sma_penalty_scale")),
        reward_far_drift_scale(settings_.getFloat("reward_far_drift_scale")),
        reward_far_drift_max(settings_.getFloat("reward_far_drift_max")),
        reward_far_angle_weight(settings_.getFloat("reward_far_angle_weight")),
        reward_near_energy_scale(settings_.getFloat("reward_near_energy_scale")),
        reward_near_energy_weight(settings_.getFloat("reward_near_energy_weight")),
        reward_dist_capture_bonus(settings_.getFloat("reward_dist_capture_bonus")),
        reward_dist_min(settings_.getFloat("reward_dist_min")),
        reward_alpha_scale(settings_.getFloat("reward_alpha_scale"))
    {
        settings.validate();
        /* Initialize random generator */
        _rng.seed(random_seed);
        sma_perturb_distrib = std::uniform_real_distribution<double>(
            -sma_perturb_max,
            sma_perturb_max
        );
        true_anomaly_distrib = std::uniform_real_distribution<double>(
            0.0,
            2 * std::numbers::pi
        );
        dist_init_offset_distrib = std::uniform_real_distribution<double>(
            capture_distance + dist_init_offset_min,
            capture_distance + dist_init_offset_max
        );
        TA_lead_distrib = std::uniform_int_distribution<int>(0, 1);

        // initialize agents: agents_states[0] = blue_sat, agents_states[1] = red_sat
        agent_ids = {"blue_sat", "red_sat"};
        agents_states.resize(num_agents);

        current_time = 0.0;

        reset();
    }

    bool OrbitalGameEnvironment::isTerminal() const
    {
        // blue_sat captured
        if (!agents_states[0].is_alive) return true;

        // red_sat fuel exhausted
        if (!agents_states[1].is_alive) return true;

        return false;
    }

    bool OrbitalGameEnvironment::isTruncated() const
    {
        return current_time >= terminal_time;
    }

    int OrbitalGameEnvironment::getObsSize(int agent_idx) const
    {
        // For both agents the observation is:
        //   [own R(3), own V(3), other's position in own LVLH frame(3), sun_direction_lvlh(3)]
        // = 12
        return 12;
    }

    double OrbitalGameEnvironment::getCurrentTime() const
    {
        return current_time;
    }

    // std::chrono::sys_time<std::chrono::milliseconds>& OrbitalGameEnvironment::getCurrentUTC() const
    // {
    //     // TODO: 完成这个函数
    //     return std::chrono::milliseconds
    // }

    void OrbitalGameEnvironment::getObservations(std::vector<Eigen::VectorXd>& observations) const
    {
        // 每个agent的observation构成如下 以下编号都是左闭右开
        // [0:6] 自身在J2000坐标下的RV
        // [6:9] 目标在LVLH坐标下的R
        // [9:12] LVLH坐标下指向太阳的单位矢量

        observations.resize(num_agents);

        // 计算当前时刻的儒略日
        auto current_utc = start_time_utc + std::chrono::seconds(static_cast<long>(current_time));
        double jd = JulianDay(std::chrono::time_point_cast<std::chrono::seconds>(current_utc));

        // 计算太阳位置
        Eigen::Vector3d sun_pos_j2000;
        solar_position(jd, sun_pos_j2000);

        // blue_sat (index 0): own RV + red_sat position in blue_sat's LVLH frame + sun angle
        {
            observations[0].resize(getObsSize(0));
            observations[0].segment<3>(0) = agents_states[0].r_j2000;
            observations[0].segment<3>(3) = agents_states[0].v_j2000;
            Eigen::Vector3d r_red_lvlh, v_red_lvlh;
            RV_J20002LVLH(
                agents_states[0].r_j2000, agents_states[0].v_j2000,
                agents_states[1].r_j2000, agents_states[1].v_j2000,
                r_red_lvlh, v_red_lvlh
            );
            observations[0].segment<3>(6) = r_red_lvlh;

            // 计算光照角（卫星到太阳的夹角）
            Eigen::Vector3d sat_to_sun = sun_pos_j2000 - agents_states[0].r_j2000;
            double cos_angle = agents_states[0].r_j2000.dot(sat_to_sun) /
                (agents_states[0].r_j2000.norm() * sat_to_sun.norm());
            observations[0](9) = std::acos(std::clamp(cos_angle, -1.0, 1.0));
        }

        // red_sat (index 1): own RV + blue_sat (target) position in red_sat's LVLH frame + sun angle
        {
            observations[1].resize(getObsSize(1));
            observations[1].segment<3>(0) = agents_states[1].r_j2000;
            observations[1].segment<3>(3) = agents_states[1].v_j2000;
            Eigen::Vector3d r_blue_lvlh, v_blue_lvlh;
            RV_J20002LVLH(
                agents_states[1].r_j2000, agents_states[1].v_j2000,
                agents_states[0].r_j2000, agents_states[0].v_j2000,
                r_blue_lvlh, v_blue_lvlh
            );
            observations[1].segment<3>(6) = r_blue_lvlh;

            // TODO： 检查光照角计算对不对
            Eigen::Vector3d sat_to_sun = sun_pos_j2000 - agents_states[1].r_j2000;
            double cos_angle = agents_states[1].r_j2000.dot(sat_to_sun) /
                (agents_states[1].r_j2000.norm() * sat_to_sun.norm());
            observations[1](9) = std::acos(std::clamp(cos_angle, -1.0, 1.0));
        }
    }

    void OrbitalGameEnvironment::getRewards(const std::vector<Eigen::Vector3d>& agent_actions,
                                            std::vector<double>& rewards) const
    {
        rewards.assign(num_agents, 0.0);
        // blue_sat reward is always 0
        // red_sat reward (index 1)
        rewards[1] += getFormationReward();
        rewards[1] += getDistanceRewardNew(1);
        rewards[1] += getCaptureReward(1);
        rewards[1] += getFuelReward(1, agent_actions[1]);
        rewards[1] += getTimeReward();
    }


    void OrbitalGameEnvironment::reset()
    {
        current_time = 0.0;
        // TODO: 这里从settings读取year month day hour minite second，然后初始化start_time_utc

        // initialize blue_sat's state
        const Eigen::Matrix<double, 6, 1> coe_base(
            settings.getFloat("sma_base", true),
            settings.getFloat("ecc_base", true),
            settings.getFloat("incl_base", true),
            settings.getFloat("RA_base", true),
            settings.getFloat("w_base", true),
            settings.getFloat("TA_base", true));
        Eigen::Matrix<double, 6, 1> coe_blue = coe_base;
        coe_blue[0] += sma_perturb_distrib(_rng);
        coe_blue[5] = true_anomaly_distrib(_rng);
        coe2rv(coe_blue, agents_states[0].r_j2000, agents_states[0].v_j2000);

        // initialize red_sat's state
        {
            Eigen::Matrix<double, 6, 1> coe_red = coe_base;
            double TA_lead = TA_lead_distrib(_rng) == 0 ? -1.0 : 1.0; // 相位超前还是滞后
            double distance_offset = dist_init_offset_distrib(_rng);
            coe_red[0] += sma_perturb_distrib(_rng);
            coe_red[5] = coe_blue[5] + TA_lead * distance_offset / coe_red[0]; // 基于blue_sat的TA加偏移
            coe2rv(coe_red, agents_states[1].r_j2000, agents_states[1].v_j2000);
        }

        // make every agent alive and reset fuel
        agents_states[0].is_alive = true;
        agents_states[0].dv_remain = dv_init_blue;
        agents_states[1].is_alive = true;
        agents_states[1].dv_remain = dv_init_red;
    }

    std::unordered_map<std::string, SatState> OrbitalGameEnvironment::getSatStates() const
    {
        std::unordered_map<std::string, SatState> result;
        for (int i = 0; i < num_agents; ++i)
        {
            result[agent_ids[i]] = agents_states[i];
        }
        return result;
    }

    void OrbitalGameEnvironment::resetWithStates(const std::unordered_map<std::string, SatState>& states)
    {
        current_time = 0.0;
        // TODO: 这里按照settings的初始时间设置
        for (int i = 0; i < num_agents; ++i)
        {
            auto it = states.find(agent_ids[i]);
            if (it != states.end())
            {
                agents_states[i] = it->second;
            }
        }
    }

    void OrbitalGameEnvironment::processDynamics(const std::vector<Eigen::Vector3d>& actions)
    {
        if (actions.size() != static_cast<size_t>(num_agents))
        {
            throw std::invalid_argument("actions.size() != num_agents");
        }

        for (int i = 0; i < num_agents; ++i)
        {
            // Apply thrust only when fuel remains; propagate orbit for all agents regardless.
            Eigen::Vector3d dv_modified = Eigen::Vector3d::Zero();
            if (agents_states[i].dv_remain > 0.0 && !almost_equal(actions[i].norm(), 0.0))
            {
                // index 0 = blue_sat, index 1 = red_sat
                double dv_max_per_step = (i == 0) ? dv_max_per_step_blue : dv_max_per_step_red;
                if (actions[i].norm() > std::min(dv_max_per_step, agents_states[i].dv_remain))
                {
                    dv_modified = std::min(dv_max_per_step, agents_states[i].dv_remain) * actions[i].normalized();
                }
                else
                {
                    dv_modified = actions[i];
                }
            }
            // 这里传入的动作是 LVLH坐标系下的，需要转换回J2000坐标系再更新 agents_states[i].v_j2000
            Eigen::Vector3d r_j2000, v_j2000;
            RV_LVLH2J2000(
                agents_states[i].r_j2000, agents_states[i].v_j2000,
                Eigen::Vector3d::Zero(), dv_modified,
                r_j2000, v_j2000
            );

            // update agent's velocity in J2000
            agents_states[i].v_j2000 = v_j2000;
            // update agent's fuel
            agents_states[i].dv_remain -= dv_modified.norm();

            // propagation
            Eigen::Vector3d r_j2000_new, v_j2000_new;
            rv_from_r0v0(agents_states[i].r_j2000, agents_states[i].v_j2000, timestep, r_j2000_new, v_j2000_new);
            agents_states[i].r_j2000 = r_j2000_new;
            agents_states[i].v_j2000 = v_j2000_new;
        }
        current_time += timestep;
    }

    void OrbitalGameEnvironment::checkAlive()
    {
        // check if blue_sat is captured by red_sat
        if ((agents_states[0].r_j2000 - agents_states[1].r_j2000).norm() < capture_distance)
        {
            agents_states[0].is_alive = false;
        }

        // 检查red_sat的燃料是否耗尽
        if (almost_equal(agents_states[1].dv_remain, 0.0))
        {
            agents_states[1].is_alive = false;
        }
    }

    double OrbitalGameEnvironment::getFormationReward() const
    {
        // Formation reward only makes sense when there are multiple red_sats.
        // With a single red_sat, this reward is always 0.
        return 0.0;
    }

    double OrbitalGameEnvironment::getDistanceRewardNew(int red_idx) const
    {
        const double distance = (agents_states[red_idx].r_j2000 - agents_states[0].r_j2000).norm();
        return -reward_phase_dist_weight * (distance - capture_distance) / capture_distance;
    }

    double OrbitalGameEnvironment::getDistanceReward(int red_idx) const
    {
        double distance = (agents_states[red_idx].r_j2000 - agents_states[0].r_j2000).norm();

        Eigen::Matrix<double, 6, 1> coe_red, coe_blue;
        rv2coe(agents_states[red_idx].r_j2000, agents_states[red_idx].v_j2000, coe_red);
        rv2coe(agents_states[0].r_j2000, agents_states[0].v_j2000, coe_blue);

        double TA_delta = std::fmod((coe_red - coe_blue)(5) + std::numbers::pi, 2.0 * std::numbers::pi) - std::numbers::pi;
        double sma_diff_ratio = (coe_red - coe_blue)(0) / coe_blue(0);

        // Far field
        double drift_product = TA_delta * sma_diff_ratio;
        double reward_far;
        if (drift_product < 0.0)
        {
            reward_far = -1.0 - std::abs(sma_diff_ratio) * reward_far_sma_penalty_scale;
        }
        else
        {
            double r_drift = std::clamp(std::abs(sma_diff_ratio) * reward_far_drift_scale, 0.0, reward_far_drift_max);
            double r_angle = (std::numbers::pi - std::abs(TA_delta)) / std::numbers::pi;
            reward_far = 1.0 * r_drift + reward_far_angle_weight * r_angle;
        }

        double dist_normalized = distance / capture_distance;
        double reward_dist = 0.0;
        if (dist_normalized <= 1.0)
        {
            reward_dist = 1.0 + reward_dist_capture_bonus * (1.0 - dist_normalized);
        }
        else if (dist_normalized <= 2.0)
        {
            reward_dist = 2.0 - dist_normalized;
        }
        else
        {
            reward_dist = std::clamp(2.0 - dist_normalized, reward_dist_min, 0.0);
        }

        double reward_energy = -std::abs(sma_diff_ratio) * reward_near_energy_scale;
        double reward_near = 1.0 * reward_dist + reward_near_energy_weight * reward_energy;

        double alpha = std::clamp(std::abs(sma_diff_ratio) * reward_alpha_scale, 0.0, 1.0);
        double total_reward = alpha * reward_far + (1.0 - alpha) * reward_near;


        return reward_phase_dist_weight * total_reward;
    }

    double OrbitalGameEnvironment::getCaptureReward(int red_idx) const
    {
        if ((agents_states[red_idx].r_j2000 - agents_states[0].r_j2000).norm() < capture_distance)
        {
            return reward_capture_weight; // capture bonus
        }

        return 0.0; // no capture
    }

    double OrbitalGameEnvironment::getFuelReward(const int red_idx, const Eigen::Vector3d& action) const
    {
        double fuel_used = action.norm();
        fuel_used = std::min(std::min(fuel_used, agents_states[red_idx].dv_remain), dv_max_per_step_red);

        return reward_fuel_weight * fuel_used;
    }

    double OrbitalGameEnvironment::getTimeReward() const
    {
        return reward_time_weight;
    }

    bool OrbitalGameEnvironment::isCaptured() const
    {
        return !agents_states[0].is_alive;
    }

    void OrbitalGameEnvironment::act(const std::vector<Eigen::Vector3d>& agents_actions)
    {
        if (agents_actions.size() != static_cast<size_t>(num_agents))
        {
            throw std::invalid_argument("agents_actions.size() != num_agents");
        }
        processDynamics(agents_actions);
        checkAlive();
    }
}
