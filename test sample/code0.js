/**
 * @license
 * Copyright Google Inc. All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.io/license
 */

import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { Comp624Component } from './comp-624.component';

describe('Comp624Component', () => {
  let component: Comp624Component;
  let fixture: ComponentFixture<Comp624Component>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ Comp624Component ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(Comp624Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
