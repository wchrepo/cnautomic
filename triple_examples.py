#%%
import re
def parse_examples_string(string_in):
    lines = string_in.splitlines()
    results = []
    for line in lines:
        line = line.partition("#")[0].strip()
        line = re.split(r"[ \t]+",line)
        results.append({'h':line[0],'r':line[1],'t':line[2]})
    return results
#%%
xWant_examples = parse_examples_string(
"""<某人X>参加聚会	xWant	喝啤酒
<某人X>大量出血	xWant	去看医生
<某人X>担任收银员	xWant	当店长
<某人X>弄脏自己	xWant	清洗干净
<某人X>熬夜学习	xWant	睡一整天
<某人X>结束一段友谊	xWant	结识新朋友
<某人X>制作自己的装束	xWant	参加化妆舞会
<某人X>打电话给<某人Y>	xWant	长聊
<某人X>告诉<某人Y>一个秘密	xWant	得到<某人Y>的建议
<某人X>修剪草坪	xWant	买一台新的割草机
<某人X>犯了一个大错	xWant	道歉
<某人X>看到<某人Y>的观点	xWant	赞同<某人Y>
<某人X>想要纹身	xWant	找到一种纹身设计
<某人X>告诉<某人Y>一些事情	xWant	得到<某人Y>的意见
<某人X>离开<某人Y>的自行车	xWant	保持自行车的安全
<某人X>拜访了一些朋友	xWant	和朋友们聊天
<某人X>坐到<某人Y>旁边	xWant	和<某人Y>成为更好的朋友
<某人X>给<某人Y>暗示	xWant	和<某人Y>约会
<某人X>买了一些爆米花	xWant	吃爆米花
<某人X>得到<某人Y>的亲笔签名	xWant	和<某人Y>建立关系
"""
)

xAttr_examples = parse_examples_string(
"""<某人X>欺负<某人Y>	xAttr	占主导地位的
<某人X>搬到另一个城市	xAttr	喜欢闯荡的
<某人X>改变了<某人Y>的想法	xAttr	雄辩的
<某人X>写一个故事	xAttr	有创造力的
<某人X>支付<某人Y>的费用	xAttr	慷慨的
<某人X>请假	xAttr	懒惰的
<某人X>向<某人Y>给出建议	xAttr	明智的
<某人X>泪流满面	xAttr	敏感的
<某人X>处理问题	xAttr	果断的
<某人X>跟踪<某人Y>	xAttr	令人害怕的
"""
)

xIntent_examples = parse_examples_string(
"""<某人X>取回报纸	xIntent	阅读报纸
<某人X>通宵工作	xIntent	赶上最后期限
<某人X>毁掉<某人Y>	xIntent	惩罚<某人Y>
<某人X>清理思绪	xIntent	准备应对新任务
<某人X>想创业	xIntent	实现自给自足
<某人X>确保<某人Y>的安全	xIntent	提供帮助
<某人X>买彩票	xIntent	变得富裕
<某人X>追球	xIntent	把球还给一些孩子
<某人X>给客服打电话	xIntent	解决一项问题
<某人X>跳来跳去	xIntent	锻炼身体
"""
)

xEffect_examples = parse_examples_string(
"""<某人X>最近离婚了	xEffect	进入了单身状态
<某人X>举重	xEffect	增加了肌肉质量
<某人X>带<某人Y>去酒吧	xEffect	喝醉了
<某人X>决定聘请家教	xEffect	学会了阅读
<某人X>给<某人Y>买礼物	xEffect	得到了<某人Y>的感谢
<某人X>听到坏消息	xEffect	晕倒了
<某人X>购买	xEffect	得到了零钱
<某人X>做了很多工作	xEffect	对工作更加擅长了
<某人X>出席音乐会	xEffect	欣赏了音乐
<某人X>履行<某人X>的职责	xEffect	收到了预期的薪水
<某人X>接到电话	xEffect	进行了一番交流
<某人X>延续<某人Y>的搜寻	xEffect	迷路了
<某人X>看到日志	xEffect	冷静了下来
"""
)

xReact_examples = parse_examples_string(
"""<某人X>与<某人Y>的家人一起生活	xReact	受到关爱
<某人X>盼望胜利	xReact	激动
<某人X>回家晚了	xReact	疲惫
<某人X>看到海豚	xReact	开心
<某人X>让<某人Y>焦虑	xReact	内疚
<某人X>破产	xReact	窘迫
<某人X>喝了一杯	xReact	神清气爽
<某人X>患有心脏病	xReact	担心自己的健康
<某人X>剃掉<某人Y>的头发	xReact	乐意帮忙
<某人X>丢失<某人Y>的所有资金	xReact	恐惧
<某人X>用完晚餐	xReact	轻松而满足
<某人X>决定去看电影	xReact	沉浸在不同的现实中
<某人X>提早回家	xReact	很高兴不用上班
<某人X>找到内心宁静	xReact	处于优雅的位置
<某人X>说了些别的	xReact	很高兴人们听了他的话
"""
)

xNeed_examples = parse_examples_string(
"""<某人X>获得工作机会	xNeed	申请
<某人X>吃东西	xNeed	处于饥饿中
<某人X>实现一次约会	xNeed	约某人出去
<某人X>结交许多新朋友	xNeed	善于交际
<某人X>改变了<某人Y>的想法	xNeed	让<某人Y>认真考虑问题
<某人X>观看视频	xNeed	拥有视频网站帐户
<某人X>玩<某人Y>的滑板	xNeed	身处滑板公园
<某人X>打个盹	xNeed	躺下
<某人X>处理情况	xNeed	遇到问题
<某人X>得到一部新手机	xNeed	为新手机付款
<某人X>把垃圾拿出去	xNeed	收集垃圾
"""
)

HinderedBy_examples = parse_examples_string(
"""<某人X>预约医生	HinderedBy	<某人X>找不到手机
<某人X>抚摸<某人Y>的额头	HinderedBy	<某人X>不敢碰<某人Y>
<某人X>吃花生酱	HinderedBy	<某人X>对花生过敏
<某人X>看上去很完美	HinderedBy	<某人X>找不到任何化妆品
<某人X>继续跑步	HinderedBy	<某人X>膝盖受伤
<某人X>与<某人Y>的家人共度时光	HinderedBy	<某人Y>的家人不喜欢与<某人X>共度时光
<某人X>从一个地方搬家到另一个地方	HinderedBy	<某人X>负担不起搬家的费用
<某人X>向政府抗议	HinderedBy	<某人X>被捕了
<某人X>大打出手	HinderedBy	<某人X>不喜欢对抗
<某人X>理解<某人Y>的感受	HinderedBy	<某人Y>不会和<某人X>交谈
<某人X>向<某人Y>提问	HinderedBy	<某人Y>听不到<某人X>说话
<某人X>提出请求	HinderedBy	<某人X>害怕被拒绝
<某人X>遇到<某人Y>的配偶	HinderedBy	<某人Y>的配偶在国外
<某人X>给水瓶加水	HinderedBy	附近没有水槽可以加水
<某人X>安排会面	HinderedBy	接待员不接<某人X>的电话
<某人X>变得富有	HinderedBy	<某人X>没有收入
<某人X>完成工作	HinderedBy	<某人X>太累无法工作
<某人X>独自坐在房间里	HinderedBy	人们进入房间打扰<某人X>
<某人X>评估<某人Y>的表演	HinderedBy	<某人X>错过了<某人Y>的表演
<某人X>割草	HinderedBy	<某人X>的割草机坏了
"""
)

default_examples = {
    "xWant":xWant_examples,
    "xAttr":xAttr_examples,
    "xIntent":xIntent_examples,
    "xEffect":xEffect_examples,
    "xReact":xReact_examples,
    "xNeed":xNeed_examples,
    "HinderedBy":HinderedBy_examples,
}

