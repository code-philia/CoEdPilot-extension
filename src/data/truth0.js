/**
 * @license
 * Copyright Google Inc. All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.io/license
 */

import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { Comp625Component } from './comp-625.component';

describe('Comp625Component', () => {
  let component: Comp625Component;
  let fixture: ComponentFixture<Comp625Component>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ Comp625Component ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(Comp625Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
