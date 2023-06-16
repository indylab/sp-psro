import json
import logging
from concurrent import futures
from typing import Tuple, List, Optional

import grpc
from google.protobuf.empty_pb2 import Empty

from grl.algos.p2sro.eval_dispatcher import EvalResultPostProcessing
from grl.algos.p2sro.p2sro_manager.p2sro_manager import P2SROManager, P2SROManagerLogger
from grl.algos.p2sro.p2sro_manager.protobuf.p2sro_manager_pb2 import NewActivePolicyRequest, PolicyMetadataRequest, \
    P2SROStatusResponse, \
    Confirmation, PolicySpecJson, NumPlayers, PayoffResult, EvalRequest, PlayerAndPolicyNum, PolicyNumList, String, \
    Metadata
from grl.algos.p2sro.p2sro_manager.protobuf.p2sro_manager_pb2_grpc import P2SROManagerServicer, \
    add_P2SROManagerServicer_to_server, \
    P2SROManagerStub
from grl.algos.p2sro.payoff_table import PayoffTable
from grl.utils.strategy_spec import StrategySpec
from grl.utils.common import SafeFallbackJSONEncoder

GRPC_MAX_MESSAGE_LENGTH = 1048576 * 40  # 40MiB

logger = logging.getLogger(__name__)


class P2SROManagerWithServer(P2SROManager):
    """
    A P2SROManager with an attached GRPC server to allow clients on other processes or computers to make calls to the
    P2SROManager's methods. Interacting with the P2SRPManager as a local instance while the server is handling requests
    is thread safe.
    """

    def __init__(self,
                 n_players,
                 is_two_player_symmetric_zero_sum: bool,
                 do_external_payoff_evals_for_new_fixed_policies: bool,
                 games_per_external_payoff_eval: int,
                 eval_dispatcher_port: int = 4536,
                 eval_result_post_processing: Optional[EvalResultPostProcessing] = None,
                 get_manager_logger=None,
                 log_dir: str = None,
                 manager_metadata: dict = None,
                 payoff_table_exponential_average_coeff: float = None,
                 port=4535,
                 restore_payoff_table_json_file_path=None):
        super(P2SROManagerWithServer, self).__init__(
            n_players=n_players,
            is_two_player_symmetric_zero_sum=is_two_player_symmetric_zero_sum,
            do_external_payoff_evals_for_new_fixed_policies=do_external_payoff_evals_for_new_fixed_policies,
            games_per_external_payoff_eval=games_per_external_payoff_eval,
            eval_dispatcher_port=eval_dispatcher_port,
            eval_result_post_processing=eval_result_post_processing,
            get_manager_logger=get_manager_logger,
            log_dir=log_dir,
            manager_metadata=manager_metadata,
            payoff_table_exponential_average_coeff=payoff_table_exponential_average_coeff,
            restore_payoff_table_json_file_path=restore_payoff_table_json_file_path
        )
        self._shutdown_error = None

        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[
            ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_LENGTH)
        ])
        servicer = _P2SROMangerServerServicerImpl(manager=self)
        add_P2SROManagerServicer_to_server(servicer=servicer, server=self._grpc_server)
        address = f'[::]:{port}'
        self._grpc_server.add_insecure_port(address)
        self._grpc_server.start()  # does not block
        logger.info(f"P2SRO Manager gRPC server listening at {address}")

    def wait_for_server_termination(self):
        self._grpc_server.wait_for_termination()
        if self._shutdown_error:
            raise self._shutdown_error

    def stop_server(self, raise_error: Exception = None):
        self._shutdown_error = raise_error
        self._grpc_server.stop(grace=2)
        if self._shutdown_error is not None:
            raise raise_error


class _P2SROMangerServerServicerImpl(P2SROManagerServicer):

    def __init__(self, manager: P2SROManagerWithServer):
        self._manager = manager

    def CheckNumPlayers(self, request: NumPlayers, context):
        try:
            is_same_num_players = self._manager.n_players() == request.num_players
            response = Confirmation(result=is_same_num_players)
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def GetLogDir(self, request, context):
        try:
            response = String(string=self._manager.get_log_dir())
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def GetManagerMetaData(self, request, context):
        try:
            json_metadata = json.dumps(self._manager.get_manager_metadata())
            response = Metadata(json_metadata=json_metadata)
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def ClaimNewActivePolicyForPlayer(self, request: NewActivePolicyRequest, context):
        try:
            policy_spec = self._manager.claim_new_active_policy_for_player(
                player=request.player,
                new_policy_metadata_dict=json.loads(s=request.metadata_json),
                set_as_fixed_immediately=request.set_as_fixed_immediately
            )
            response = PolicySpecJson(policy_spec_json=policy_spec.to_json())
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def SubmitNewActivePolicyMetadata(self, request: PolicyMetadataRequest, context):
        try:
            policy_spec = self._manager.submit_new_active_policy_metadata(
                player=request.player,
                policy_num=request.policy_num,
                metadata_dict=json.loads(s=request.metadata_json)
            )
            response = PolicySpecJson(policy_spec_json=policy_spec.to_json())
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def CanActivePolicyBeSetAsFixedNow(self, request: PlayerAndPolicyNum, context):
        try:
            can_be_fixed_now = self._manager.can_active_policy_be_set_as_fixed_now(
                player=request.player,
                policy_num=request.policy_num
            )
            response = Confirmation(result=can_be_fixed_now)
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def SetActivePolicyAsFixed(self, request: PolicyMetadataRequest, context):
        try:
            policy_spec = self._manager.set_active_policy_as_fixed(
                player=request.player,
                policy_num=request.policy_num,
                final_metadata_dict=json.loads(s=request.metadata_json)
            )
            response = PolicySpecJson(policy_spec_json=policy_spec.to_json())
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def GetCopyOfLatestData(self, request, context):
        try:
            payoff_able, active_policies_per_player, fixed_policies_per_player = self._manager.get_copy_of_latest_data()
            response = P2SROStatusResponse(payoff_table_json=payoff_able.to_json_string())

            for player_active_polices, player_fixed_policies in (
                    zip(active_policies_per_player, fixed_policies_per_player)):
                active_list = PolicyNumList()
                fixed_list = PolicyNumList()
                active_list.policies.extend(player_active_polices)
                fixed_list.policies.extend(player_fixed_policies)
                response.active_policies.append(active_list)
                response.fixed_policies.append(fixed_list)

        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def SubmitEmpiricalPayoffResult(self, request: PayoffResult, context):
        try:
            policy_specs_for_each_player = tuple(StrategySpec.from_json(json_string=json_string)
                                                 for json_string in request.json_policy_specs_for_each_player)
            # noinspection PyTypeChecker
            self._manager.submit_empirical_payoff_result(
                policy_specs_for_each_player=policy_specs_for_each_player,
                payoffs_for_each_player=request.payoffs_for_each_player,
                games_played=request.games_played,
                override_all_previous_results=request.override_all_previous_results
            )
            response = Confirmation(result=True)
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def RequestExternalEval(self, request: EvalRequest, context):
        try:
            policy_specs_for_each_player = tuple(StrategySpec.from_json(json_string=json_string)
                                                 for json_string in request.json_policy_specs_for_each_player)
            self._manager.request_external_eval(policy_specs_for_each_player=policy_specs_for_each_player)
            response = Confirmation(result=True)
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response

    def IsPolicyFixed(self, request: PlayerAndPolicyNum, context):
        try:
            is_policy_fixed = self._manager.is_policy_fixed(
                player=request.player,
                policy_num=request.policy_num
            )
            response = Confirmation(result=is_policy_fixed)
        except Exception as e:
            self._manager.stop_server(raise_error=e)
            raise e
        return response


class RemoteP2SROManagerClient(P2SROManager):
    """
    GRPC client for a P2SROManager server.
    Behaves exactly like a local P2SROManager but actually is connecting to a remote P2SRO Manager on another
    process or computer.
    """

    # noinspection PyMissingConstructor
    def __init__(self, n_players: int, port: int = 4535, remote_server_host: str = "127.0.0.1"):
        self._stub = P2SROManagerStub(channel=grpc.insecure_channel(target=f"{remote_server_host}:{port}", options=[
            ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_LENGTH),
        ]))
        server_is_same_num_players: Confirmation = self._stub.CheckNumPlayers(NumPlayers(num_players=n_players))
        if not server_is_same_num_players.result:
            raise ValueError("Remote P2SROManger server has a different number of players than this one.")
        self._n_players = n_players

    def n_players(self) -> int:
        return self._n_players

    def get_log_dir(self) -> str:
        return self._stub.GetLogDir(Empty()).string

    def get_manager_metadata(self) -> dict:
        response: Metadata = self._stub.GetManagerMetaData(Empty())
        return json.loads(response.json_metadata)

    def claim_new_active_policy_for_player(self, player, new_policy_metadata_dict, set_as_fixed_immediately=False) -> StrategySpec:
        try:
            metadata_json = json.dumps(obj=new_policy_metadata_dict, cls=SafeFallbackJSONEncoder)
        except (TypeError, OverflowError) as json_err:
            raise ValueError(f"new_policy_metadata_dict must be JSON serializable."
                             f"When attempting to serialize, got this error:\n{json_err}")
        request = NewActivePolicyRequest(
            player=player,
            metadata_json=metadata_json,
            set_as_fixed_immediately=set_as_fixed_immediately
        )
        response: PolicySpecJson = self._stub.ClaimNewActivePolicyForPlayer(request)
        return StrategySpec.from_json(response.policy_spec_json)

    def submit_new_active_policy_metadata(self, player, policy_num, metadata_dict) -> StrategySpec:
        try:
            metadata_json = json.dumps(obj=metadata_dict, cls=SafeFallbackJSONEncoder)
        except (TypeError, OverflowError) as json_err:
            raise ValueError(f"metadata_dict must be JSON serializable."
                             f"When attempting to serialize, got this error:\n{json_err}")
        request = PolicyMetadataRequest(
            player=player,
            policy_num=policy_num,
            metadata_json=metadata_json
        )
        response: PolicySpecJson = self._stub.SubmitNewActivePolicyMetadata(request)
        return StrategySpec.from_json(response.policy_spec_json)

    def can_active_policy_be_set_as_fixed_now(self, player, policy_num) -> bool:
        response: Confirmation = self._stub.CanActivePolicyBeSetAsFixedNow(PlayerAndPolicyNum(player=player,
                                                                                              policy_num=policy_num))
        return response.result

    def set_active_policy_as_fixed(self, player, policy_num, final_metadata_dict) -> StrategySpec:
        try:
            metadata_json = json.dumps(obj=final_metadata_dict, cls=SafeFallbackJSONEncoder)
        except (TypeError, OverflowError) as json_err:
            raise ValueError(f"final_metadata_dict must be JSON serializable."
                             f"When attempting to serialize, got this error:\n{json_err}")
        request = PolicyMetadataRequest(
            player=player,
            policy_num=policy_num,
            metadata_json=metadata_json
        )
        response: PolicySpecJson = self._stub.SetActivePolicyAsFixed(request)
        return StrategySpec.from_json(response.policy_spec_json)

    def get_copy_of_latest_data(self) -> (PayoffTable, List[List[int]], List[List[int]]):
        request = Empty()
        response: P2SROStatusResponse = self._stub.GetCopyOfLatestData(request)
        payoff_table = PayoffTable.from_json_string(json_string=response.payoff_table_json)

        active_policies_per_player = []
        fixed_policies_per_player = []
        for active_list, fixed_list in zip(response.active_policies, response.fixed_policies):
            active_policies_per_player.append(list(active_list.policies))
            fixed_policies_per_player.append(list(fixed_list.policies))

        return payoff_table, active_policies_per_player, fixed_policies_per_player

    def submit_empirical_payoff_result(self,
                                       policy_specs_for_each_player: Tuple[StrategySpec],
                                       payoffs_for_each_player: Tuple[float],
                                       games_played: int,
                                       override_all_previous_results: bool):
        request = PayoffResult(games_played=games_played, override_all_previous_results=override_all_previous_results)
        request.json_policy_specs_for_each_player.extend(spec.to_json() for spec in policy_specs_for_each_player)
        request.payoffs_for_each_player.extend(payoffs_for_each_player)
        self._stub.SubmitEmpiricalPayoffResult(request)

    def request_external_eval(self, policy_specs_for_each_player):
        request = EvalRequest()
        request.json_policy_specs_for_each_player.extend(spec.to_json() for spec in policy_specs_for_each_player)
        self._stub.RequestExternalEval(request)

    def is_policy_fixed(self, player, policy_num):
        response: Confirmation = self._stub.IsPolicyFixed(PlayerAndPolicyNum(player=player, policy_num=policy_num))
        return response.result


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    manager = P2SROManagerWithServer(
        n_players=2,
        is_two_player_symmetric_zero_sum=True,
        do_external_payoff_evals_for_new_fixed_policies=True,
        games_per_external_payoff_eval=2000,
        payoff_table_exponential_average_coeff=0.001
    )
    print(f"Launched P2SRO Manager with server.")
    manager.wait_for_server_termination()
