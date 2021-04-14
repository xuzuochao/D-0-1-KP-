# -*- coding:utf-8 -*-
import os
import wx
class MyFrame(wx.Frame):
    def __init__(self,parent,id):
        wx.Frame.__init__(self, parent,id, title="首页",size=(600,450))
        #创建面板
        panel = wx.Panel(self) 
        # 创建文本和输入框
        self.title1 = wx.StaticText(panel ,label="D{0-1}KP 实例数据集算法实验平台",pos=(60,20))
     
        #font  = wx.Font(14, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        font  = wx.Font(16, wx.SWISS, wx.ITALIC, wx.LIGHT)
        self.title1.SetFont(font)
        
        self.bt_con  = wx.Button(panel,label='登录/注册',pos=(500,20))
        self.bt_con.Bind(wx.EVT_BUTTON,self.OnclickSubmit)
    
        self.bt_confirm = wx.Button(panel,label='首     页',pos=(0,90))
        self.bt_cancel  = wx.Button(panel,label='查看数据',pos=(0,115))
        self.bt_cancel  = wx.Button(panel,label='动态规划',pos=(0,140))
        self.bt_cancel  = wx.Button(panel,label='回 溯 法',pos=(0,165))
        self.bt_cancel  = wx.Button(panel,label='遗传算法',pos=(0,190))
        self.bt_cancel  = wx.Button(panel,label='降序排列',pos=(0,215))
        self.bt_cancel  = wx.Button(panel,label='画散点图',pos=(0,235))
        
        self.title2 = wx.StaticText(panel ,label="联系我们",pos=(230,285))
        font1  = wx.Font(14, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        #font  = wx.Font(16, wx.SWISS, wx.ITALIC, wx.LIGHT)
        self.title2.SetFont(font1)
        self.title = wx.StaticText(panel ,label="电子邮件：1119786516@qq.com",pos=(180,315))
        self.title = wx.StaticText(panel ,label="地址：西北师范大学",pos=(200,340))
        self.title = wx.StaticText(panel ,label="邮政编码",pos=(230,365))
        self.title = wx.StaticText(panel ,label="联系电话",pos=(230,390))

    def OnclickSubmit(self,event):
        """ 点击确定按钮，执行方法 """
        def fun():
            os.system("登录.py")
        fun()

if __name__ == '__main__':        
    app = wx.App()                      # 初始化
    frame = MyFrame(parent=None,id=-1)  # 实例MyFrame类，并传递参数    
    frame.Show()                        # 显示窗口
    app.MainLoop()                      # 调用主循环方法
