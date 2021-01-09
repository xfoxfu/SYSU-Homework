#ifndef REFUND_H
#define REFUND_H

#include "mysqlclient.h"

/**
 * 退货
 * @param order_id 购书订单ID
 * @param count 退货数量
 * @return refund_id, 失败返回-1
 */
int refund(MySQLClient &client, int order_id, int count);

/**
 * 测试refund函数
 */
void testRefund(MySQLClient &client);

#endif