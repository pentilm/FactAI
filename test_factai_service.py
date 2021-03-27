import sys
import grpc

# import the generated classes
import service.service_spec.factai_service_pb2_grpc as pb2_grpc
import service.service_spec.factai_service_pb2 as pb2

from service import registry

if __name__ == "__main__":
	try:
		test_flag = False
		if len(sys.argv) == 2:
			if sys.argv[1] == "auto":
				test_flag = True

		# Fact AI Service - Arithmetic
		endpoint = input("Endpoint (localhost:{}): ".format(registry["factai_service"]["grpc"])) if not test_flag else ""
		if endpoint == "":
			endpoint = "localhost:{}".format(registry["factai_service"]["grpc"])

		grpc_method = "stance_classify"

		headline = 'Melania Trump cancels plans to attend Tuesday rally citing Covid recovery'
		body = '''Melania Trump is canceling her first campaign appearance in
		months because she is not feeling well as she continues to recover from
		Covid-19.  She had been set to join President Donald Trump's rally in
		Pennsylvania on Tuesday night, but she has decided not to go. "Mrs. Trump
		continues to feel better every day following her recovery from Covid-19, but
		with a lingering cough, and out of an abundance of caution, she will not be
		traveling today," said Stephanie Grisham, the first lady's chief of staff.
		It would have been the first lady's first in-person appearance at a campaign
		event, outside of August's Republican National Convention speech at the
		White House, in more than a year, when she joined the President last June at
		the official reelection kick-off rally in Florida. Trump wrote an essay last
		week that her symptoms of Covid-19, "hit me all at once and it seemed to be
		a roller coaster." She described having body aches, a cough, headaches and
		feeling extreme fatigue. There are no plans for Melania Trump to make up for
		the rally, according to a source familiar with the first lady's schedule.
		The first lady "did not offer options for another campaign appearance, at a
		rally or otherwise. Let's put it this way, there was no discussion of a rain
		date," the source told CNN. Melania Trump's health issues are "genuine" and
		"she has a persistent cough," they added. "This is not the time to stay
		completely out of the spotlight," the source added. The first lady was not
		expected to give solo remarks at Tuesday's Pennsylvania event. Travel
		restrictions due to coronavirus throughout the last several months, and her
		own bout with Covid-19, hindered Trump's work schedule, Grisham told CNN.
		However, the President, vice president and other members of the Trump family
		have hit the campaign trail. Melania Trump has not historically been a
		visible campaign presence, eschewing appearances while other Trump
		surrogates crisscross the country. In the entire 2016 election cycle, Trump
		gave only a handful of solo speeches. Her longest in Pennsylvania, was just
		five days before the 2016 election, and came after a months' long hiatus
		from the campaign trail. "I'm an immigrant, and let me tell you that nobody
		values the freedom and opportunity of America more than me," said Trump at
		the time, after an introduction by second lady Karen Pence, an active
		campaigner in both 2016 and 2020. In 2016, the most the public saw of their
		future first lady was her presence at the presidential debates, something
		Trump is doing this time around, as well. She attended Trump's debate
		against Democratic rival Joe Biden in Ohio, and she is expected to attend
		the final 2020 presidential debate on Thursday in Nashville, Tennessee. If
		2016 showed a hesitant potential first-spouse, 2020 is proving Trump remains
		ambivalent to the barnstorming ways of her stepchildren, all of whom have
		been hosting campaign events for the last several weeks in key battleground
		states. With few big-name Republican surrogates outside of his family, and
		lack of participation from his wife, Trump's adult children are doing most
		of the talking to voters, with the President himself committing in recent
		days to two to three rallies in a 24-hour period. This week, Ivanka Trump
		will make stops in Michigan, Wisconsin, North Carolina and Florida; Eric
		Trump heads to New Hampshire and Michigan, while his wife, Lara Trump, a
		member of the reelection campaign, goes to Nevada and Arizona; Donald Trump
		Jr. will be at events in North Carolina and Pennsylvania.'''

		# Open a gRPC channel
		channel = grpc.insecure_channel("{}".format(endpoint))
		stub = pb2_grpc.FACTAIStanceClassificationStub(channel)
		inp = pb2.InputData(headline=headline, body=body)

		if grpc_method == "stance_classify":
			response = stub.stance_classify(inp)
			print(response)
		else:
			print("Invalid method!")
			exit(1)
			
	except Exception as e:
		print(e)
		exit(1)
