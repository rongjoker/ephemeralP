---
update
    bas_location_bk bk
  left join bas_warehouse_cd bc
 on bk.cd_warehouse_code = bc.cd_warehouse_code
  set bk.bas_zone_id = 1, bk.bas_zone_code = 'LC_YL_HC'
 where bk.cd_warehouse_code is not null and bk.bas_zone_id is null
 and bc.warehouse_code = 'BHLC'
 ;

  update
    bas_location_bk bk
  left join bas_warehouse_cd bc
 on bk.cd_warehouse_code = bc.cd_warehouse_code
  set bk.bas_zone_id = 6, bk.bas_zone_code = 'XC_YL_HC'
 where bk.cd_warehouse_code is not null and bk.bas_zone_id is null
 and bc.warehouse_code = 'BHXC'
 ;


 -- 1846
select count(1) from bas_location_bk;
-- 2678
-- 3726

